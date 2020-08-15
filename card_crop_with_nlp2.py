
import numpy as np
import cv2
from PIL import Image, ImageDraw

from scipy.ndimage.filters import rank_filter
import builtins
import pytesseract
import re
import nltk
import sys
from unidecode import unidecode
from geotext import GeoText
import zipcode
import json
original_open = open
from nltk.corpus import stopwords
stop = stopwords.words('english')



def bin_open(filename, mode='rb'):
    return original_open(filename, mode)


def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[(int)((N - 1) / 2), :] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations * 2)

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[:, (int)((N - 1) / 2)] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations - 1)

    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0)) / 255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.

    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 50
    dilation = 5
    n = 2
    while count > 6:
        n += 2
        # print(edges.dtype)
        dilated_image = dilate(edges, N=5, iterations=n)
        #cv2.imshow('dilate', dilated_image)
        #
        # print(dilated_image.dtype)

        dilated_image = dilated_image.astype(np.uint8)
        # print(dilated_image.dtype)

        _, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        count = len(contours)
    # print dilation
    # Image.fromarray(edges).show()
    # Image.fromarray(255 * dilated_image).show()
    return contours


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.

    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        # print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                            remaining_frac > 0.25 and new_area_frac < 0.15):
                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                    i, covered_sum, new_sum, total, remaining_frac,
                    crop_area(crop), crop_area(new_crop), area, new_area_frac,
                    f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.

    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop

    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im


def max1(a, b):
    if a > b:
        return a
    else:
        return b


def min1(a, b):
    if a <= b:
        return a
    else:
        return b


def process_image(quad,mode):
    #orig_im = Image.open(path)
    #scale, im = downscale_image(orig_im)
    #im = np.asarray(im)
    im = quad
    img1 = quad

    # cv2.imshow('image',im)
    # 


    edges = cv2.Canny(im, 100, 200)

    # print (edges.dtype)

    # TODO: dilate image _before_ finding a border. This is crazy sensitive!

    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    borders.sort(key=lambda iabcd: (iabcd[3] - iabcd[1]) * (iabcd[4] - iabcd[2]))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    #cv2.imshow('edges', edges)
    #
    contours = find_components(edges)
    if len(contours) == 0:
        #print('%s -> (no text!)' % path)
        return

    c = ""
    q=""
    l=[]
    m=[]
    if mode==0:
      for i in range(len(contours)):
     
        x, y, w, h = cv2.boundingRect(contours[i])
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(_, [box], 0, (0, 0, 255), 1)
        # cv2.drawContours(img, [contours[i]], -1, (255, 0, 0), 2)
        if w > 150 and h > 30:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi = img1[y:y + h, x:x + w]
            nimg = cv2.bilateralFilter(roi,9,3,3)
            nimg = cv2.cvtColor(nimg,cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            nimg = cv2.morphologyEx(nimg,cv2.MORPH_OPEN,kernel)
            nimg = cv2.adaptiveThreshold(nimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,99,14)
            imaga = Image.fromarray(nimg)
            bts = pytesseract.image_to_string(imaga)
            q=q+bts
            l.append(bts)
            m.append([bts,x,y])
        m.sort(key=lambda item:item[2])
        m.sort(key=lambda item:item[1])     
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kernel_sharpen_3 = np.array(
            [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0
        #im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 14)
        im = cv2.filter2D(im, -1, kernel_sharpen_3)
        imaga = Image.fromarray(im)
        bts = pytesseract.image_to_string(imaga)
        
        #print (bts)
        c=c+bts
        l.append(bts)
        #print (l)
    #cv2.imshow('image2', im)
    for i in m:
        c = c+ i[0]+"\n"
    #print (l)
    
    y=[]
    for i in l:
        i=i.replace("\n"," ")
        y.append(i)
        
    #print ("TESSERACT-OUTPUT:")
    #print (c)
    o="&!*#<>|&%#"
    for i in o:
        c=c.replace(i,'')
    print(c)
    jsonoutput={}
    numbers = extract_phone_numbers(c)
    emails = extract_email_addresses(c)
    #lol=wt(emails[0])
    #print(lol)
   
    #address=extract_address(c)
    
    website=extract_website(c)
    st=removenum(c,numbers)
    if (extract_zipcode(st))is not None:
        #print(extract_zipcode(st))
        zipcode=unidecode(extract_zipcode(st))
    else:
        zipcode=""
    
    co=extract_country(c)
    #print(co)
    if co is not None:
       country,country1=co[0],co[1]
    else:
        country=None
        country1=None
    org1=organise(c)
    state=extract_state(zipcode)
    cities=extract_cities(c)
    address=extract_address(country,state,cities,zipcode,y)
    if emails!=[] and address is not None:
      address=address.replace(emails[0],'')
    modifiedaddress=remove_et(address,emails,numbers)
    matchnumbers=match(numbers,c)
    reducestring=remove_un(c,address,emails,numbers,website)
    
    organisations = extract_organisation(reducestring)
    designations=extract_designations(reducestring)
    #print(designations)
    
    if address is not None: 
      word=wt(address)
      j=len(word)
      for k in range(j):
        for i in word[k]:
          reducestring=reducestring.replace(i,'')

        
    names = extract_names(reducestring)
    namematch=name_match(reducestring,designations,names)  
    if emails!=[]:
       jsonoutput["emails"]=emails[0]
    else:
        jsonoutput["emails"]=None
    jsonoutput["website"]=website
    if cities!=[]:
       jsonoutput["cities"]=cities[0]
    else:
        jsonoutput["cities"]=None
    jsonoutput["zipcode"]=zipcode
    jsonoutput["state"]=state
    jsonoutput["country"]=country1
    jsonoutput["address"]=modifiedaddress
    
    jsonoutput["designations"]=designations
    jsonoutput["organsiations"]=org1
    #if numbers!=[]:
        #jsonoutput["numbers"]=numbers
    a=['Office','Off','(O)','Telephone','Phone','phone','PHONE','Ph','P:','(P)' ,'Tel','tel','p','P','T']
    b=['Mobile','mobile','Mob','Cell','Main','(M)']
    f=['Fax','fax','FAX','F:','(F)','Fax','F','f']
    d=['Direct','Dir','Direct Dial','Dial','D','toll free']
    if matchnumbers!=[]:
       #jsonoutput["matchnumbers"]=matchnumbers
       for i in matchnumbers:
           for j in range(len(a)):
               if a[j] in i:
                   x=i
                   x=x.replace(a[j],'')
                   jsonoutput["phone"]=x
                   break
           for j in range(len(b)):
                if b[j] in i:
                    x=i
                    x=x.replace(b[j],'')
                    jsonoutput["mobile"]=x
                    break
           for j in range(len(f)):
               if f[j] in i:
                   x=i
                   x=x.replace(f[j],'')
                   jsonoutput["fax"]=x
                   break
           for j in range(len(d)):
               if d[j] in i:
                   x=i
                   x=x.replace(d[j],'')
                   jsonoutput["direct dial"]=x
                   break
                
            
            
       
    #if names!=[]:
        #jsonoutput["names"]=names
    jsonoutput["firstname"]=None
    jsonoutput["lastname"]=None
    if namematch!=[]:
        for i in range(len(namematch)):
            if i==0 :
                jsonoutput["firstname"]=namematch[i]
            
            if i==1:
                jsonoutput["lastname"]=namematch[i]
    else:
        if emails!=[]:
            s=""
            for i in emails[0]:
                if i=='@':
                    break
                else:
                    s+=i
            jsonoutput["firstname"]=s
                
                
    #print("jsonoutput:",jsonoutput)
    print("jsonstring:",json.dumps(jsonoutput))
    #print ("address:",address)
    print("reducedstring:",reducestring)
    '''print("FIELD_MATCHING:")
    print ("numbers:", numbers)
    print("matchnumbers:",matchnumbers)
    print ("email:",emails)
    print("organisations:",organisations)
    print("names:",names)
    
    print("modified address:",modifiedaddress)
    print ("city:",cities)
    print("website:",website)
    print("country:",country)
    print("zipcode:",zipcode)
    print ("state:",state)
    print ("organisation:",org1)
    print("designations:",designations)
    
    '''
def extract_phone_numbers(string):
    r = re.compile(r'.*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?')
    phone_numbers = r.findall(string)
    return [number for number in phone_numbers]
def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)
def extract_cities(string):
    places=GeoText(string)
    return (places.cities)
def extract_website(string):
    r=re.compile(r'[\w\.-]+\.com')
    website=r.findall(string)
    #print (website)
    for i in website:
        if '@' not in i :
            s=i[:3]
            if s=='www':
               return (i)
def organise(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    x= r.findall(string)
    s=""
   # print (x)
    if x!=[]:
       s=x[0]
    
       s=s[:len(s)-4]
       for k,i in enumerate(s):
          if i=='@':
             break
       s=s[k+1:]
    return (s)
def extract_state(string):
    #rint (string)
    if string!="":
      zip_=zipcode.isequal(string)
      if zip_ is None:
        return None
      else:
        return (zip_.state)
    else:
        return None

def extract_address(country,state,city,zipcode,l):
   #print("address:")
    #print (country,state,city,zipcode)
    for i in l:
        if country is not None and state is not None and city!=[] and zipcode is not None:
           if (country in i  )or (state in i) or (city[0] in i) or (zipcode in i):
               #print (i)
               return (i)
        elif country is  None and state is not None and city!=[] and zipcode is not None:
           if  (state in i) or (city[0] in i) or (zipcode in i):
               #print (i)
               return (i)
        elif country is  None and state is  None and city!=[] and zipcode is not None:
           if  (city[0] in i) or (zipcode in i):
               #print (i)
               return (i)
        elif country is not None and state is  None and city!=[] and zipcode is not None:
           if (country in i  ) or (city[0] in i) or (zipcode in i):
               #print (i)
               return (i)
        elif country is not None and state is  None and city==[] and zipcode is not None:
           if (country in i  )  or (zipcode in i):
               #print (i)
               return (i)
                
          
def remove_et(address,email,number):
   #print("modified address:")
   y=""
   y=address
   #print (email,address)
   if email!=[] and address is not None:
    if email[0] in address:
      y=address.replace(email[0],'')
     #print (y)
    for  i in number:
       if i in y:
           y=y.replace(i,'')
   #print (y)
    a=['Office','Off','Main','Telephone','Tel','Direct','Dir','Phone','Ph','Mobile','Mob','Cell','Fax','fax','direct','mobile','toll free','tel']
    for i in a:
       if i in y:
           y=y.replace(i,'')
   #print (y)
    
   #print (y)
   return y
def removenum(c,number):
    y=""
    y=c
    for  i in number:
       if i in y:
           y=y.replace(i,'')
    return y
def match(numbers,c):
    #print("match numbers:")
    a=['Office','Off','Main','Telephone','Direct','Dir','Phone','phone','PHONE','Ph','Mobile','mobile','Mob','Cell','Fax','fax','FAX','P:','F:','M:','Direct Dial',
       '(P)','(F)','(M)','Tel','Fax','(O)','Dial','toll free','T','F','tel','D','p','f']
    l=[]
    #print(len(numbers))
    for i in range(len(numbers)):
       
        flag=1
        f=c.find(numbers[i])
        #print (i,f)
        #if k==0:
        m=c[f-12:f]
        for j in a:
            s=""
            if j in m:
                flag=0
                #print(j,i)
                s=s+j+numbers[i]
                l.append(s)
                #x=len(j)
                #y=f+x
                #print (y)
           
                if j=='Office' or j=='Off' or j=='(O)': 
                   a.remove('Office')
                   a.remove('Off')
                   a.remove('(O)')
                   break
                elif j=='Main' or j=='(M)':
                   a.remove('Main')
                   a.remove('(M)')
                   break
                elif j=='Phone' or j=='PHONE' or j=='ph' or j=='phone' or j=='P:' or j=='(P)'or j=='Telephone' or j=='Tel' or j=='tel' or j=='p':
                   a.remove('Phone')
                   a.remove('tel')
                   a.remove('PHONE')
                   a.remove('phone')
                   a.remove('P:')
                   a.remove('p')
                   a.remove('(P)')
                   a.remove('Telephone')
                   a.remove('Tel')
                   break 
                elif j=='Direct' or j=='Dir' or j=='Direct Dial' or j=='Dial' or j=='D':
                    a.remove('Direct')
                    a.remove('Dir')
                    a.remove('Direct Dial')
                    a.remove('Dial')
                    a.remove('D')
                    break
                elif j=='Fax' or j=='FAX' or j=='fax' or j=='F:' or j=='(F)' or j=='f':
                    a.remove('Fax')
                    a.remove('FAX')
                    a.remove('fax')
                    a.remove('F:')
                    a.remove('f')
                    a.remove('(F)')
                    break
                elif j=='Mobile' or j=='Mob' or j=='mobile' or j=='Cell':
                     a.remove('Mobile')
                     a.remove('mobile')
                     a.remove('Mob')
                     a.remove('Cell')
                     break
                elif j=='toll free':
                    a.remove('toll free')
                    break
    return (l)
def  remove_un(c,address,emails,numbers,website):
 
    x=c
    #a=['Main','Telephone','Direct','Dir','Phone','phone','PHONE','Ph','Mobile','mobile','Mob','Cell','Fax','fax','FAX','Direct Dial',':','E-mail','E-MAIL',
       #'Dial','(P)','(F)','(M)']
    if address is not None:
       x=x.replace(address,'')
    if emails!=[]:
       x=x.replace(emails[0],'')
    for i in numbers:
       if i is not None:
         x=x.replace(i,'')
    if website is not None:
      x=x.replace(website,'')
    ''' for i in a:
        if i in x:
            x=x.replace(i,'')'''
    return (x)
def extract_designations(c):
    s=""
    y=c.lower()
    #print (y)
    l=['Certified','Chartered','Fellow','Member','Doctor','Head','Senior','Vice','Chief' ,'Administrative','Controller','Clerk','Specialist','General',
       'Director',' Founder','Founding', 'Advisor','Executive','Professor', 'Counsel','Managing',' MD',' VP',' VC','CRE',
       'Associate','Principal','Junior ', 'Assistant','Representative','Regional','Specialist','Coordinator','Deputy','Financial','Operating','Branch']
    for i in l:
        x=i.lower()
        
        if x in y:
            s+=i+" "
    
    m=['President','Partner', 'Manager','Officer','Editor','Publisher','Chairman','Lead','Leader','Proprietor','Consultant','Attorney','Director']
    for i in m:
        x=i.lower()
        if x in y:
            s+=i+" "
    o=['&','and','of']
    for i in o:
        #x=i.lower()
        
        if i in c:
            s+=i+" "
    
    p=['CEO','CFO',' CTO',' COO','Agent','Architect','Engineer','Scientist','Lawyer','Surgeon','Physician','Instructor','Accountant','Auditor', 'Consultant','Analyst',
       'Librarian','Barrister','Researcher' ,'Scholar','Trainer','Technisan','Therapist','Finance']
    for i in p:
        x=i.lower()
        if x in y:
            s+=i+" "
 
   
    return (s)


    
        
def extract_country(string):
    v= [
    ('US', 'United States'),
    ('AF', 'Afghanistan'),
    ('AL', 'Albania'),
    ('DZ', 'Algeria'),
    ('AS', 'American Samoa'),
    ('AD', 'Andorra'),
    ('AO', 'Angola'),
    ('AI', 'Anguilla'),
    ('AQ', 'Antarctica'),
    ('AG', 'Antigua And Barbuda'),
    ('AR', 'Argentina'),
    ('AM', 'Armenia'),
    ('AW', 'Aruba'),
    ('AU', 'Australia'),
    ('AT', 'Austria'),
    ('AZ', 'Azerbaijan'),
    ('BS', 'Bahamas'),
    ('BH', 'Bahrain'),
    ('BD', 'Bangladesh'),
    ('BB', 'Barbados'),
    ('BY', 'Belarus'),
    ('BE', 'Belgium'),
    ('BZ', 'Belize'),
    ('BJ', 'Benin'),
    ('BM', 'Bermuda'),
    ('BT', 'Bhutan'),
    ('BO', 'Bolivia'),
    ('BA', 'Bosnia And Herzegowina'),
    ('BW', 'Botswana'),
    ('BV', 'Bouvet Island'),
    ('BR', 'Brazil'),
    ('BN', 'Brunei Darussalam'),
    ('BG', 'Bulgaria'),
    ('BF', 'Burkina Faso'),
    ('BI', 'Burundi'),
    ('KH', 'Cambodia'),
    ('CM', 'Cameroon'),
    ('CA', 'Canada'),
    ('CV', 'Cape Verde'),
    ('KY', 'Cayman Islands'),
    ('CF', 'Central African Rep'),
    ('TD', 'Chad'),
    ('CL', 'Chile'),
    ('CN', 'China'),
    ('CX', 'Christmas Island'),
    ('CC', 'Cocos Islands'),
    ('CO', 'Colombia'),
    ('KM', 'Comoros'),
    ('CG', 'Congo'),
    ('CK', 'Cook Islands'),
    ('CR', 'Costa Rica'),
    ('CI', 'Cote D`ivoire'),
    ('HR', 'Croatia'),
    ('CU', 'Cuba'),
    ('CY', 'Cyprus'),
    ('CZ', 'Czech Republic'),
    ('DK', 'Denmark'),
    ('DJ', 'Djibouti'),
    ('DM', 'Dominica'),
    ('DO', 'Dominican Republic'),
    ('TP', 'East Timor'),
    ('EC', 'Ecuador'),
    ('EG', 'Egypt'),
    ('SV', 'El Salvador'),
    ('GQ', 'Equatorial Guinea'),
    ('ER', 'Eritrea'),
    ('EE', 'Estonia'),
    ('ET', 'Ethiopia'),
    ('FK', 'Falkland Islands (Malvinas)'),
    ('FO', 'Faroe Islands'),
    ('FJ', 'Fiji'),
    ('FI', 'Finland'),
    ('FR', 'France'),
    ('GF', 'French Guiana'),
    ('PF', 'French Polynesia'),
    ('TF', 'French S. Territories'),
    ('GA', 'Gabon'),
    ('GM', 'Gambia'),
    ('GE', 'Georgia'),
    ('DE', 'Germany'),
    ('GH', 'Ghana'),
    ('GI', 'Gibraltar'),
    ('GR', 'Greece'),
    ('GL', 'Greenland'),
    ('GD', 'Grenada'),
    ('GP', 'Guadeloupe'),
    ('GU', 'Guam'),
    ('GT', 'Guatemala'),
    ('GN', 'Guinea'),
    ('GW', 'Guinea-bissau'),
    ('GY', 'Guyana'),
    ('HT', 'Haiti'),
    ('HN', 'Honduras'),
    ('HK', 'Hong Kong'),
    ('HU', 'Hungary'),
    ('IS', 'Iceland'),
    ('IN', 'India'),
    ('ID', 'Indonesia'),
    ('IR', 'Iran'),
    ('IQ', 'Iraq'),
    ('IE', 'Ireland'),
    ('IL', 'Israel'),
    ('IT', 'Italy'),
    ('JM', 'Jamaica'),
    ('JP', 'Japan'),
    ('JO', 'Jordan'),
    ('KZ', 'Kazakhstan'),
    ('KE', 'Kenya'),
    ('KI', 'Kiribati'),
    ('KP', 'Korea (North)'),
    ('KR', 'Korea (South)'),
    ('KW', 'Kuwait'),
    ('KG', 'Kyrgyzstan'),
    ('LA', 'Laos'),
    ('LV', 'Latvia'),
    ('LB', 'Lebanon'),
    ('LS', 'Lesotho'),
    ('LR', 'Liberia'),
    ('LY', 'Libya'),
    ('LI', 'Liechtenstein'),
    ('LT', 'Lithuania'),
    ('LU', 'Luxembourg'),
    ('MO', 'Macau'),
    ('MK', 'Macedonia'),
    ('MG', 'Madagascar'),
    ('MW', 'Malawi'),
    ('MY', 'Malaysia'),
    ('MV', 'Maldives'),
    ('ML', 'Mali'),
    ('MT', 'Malta'),
    ('MH', 'Marshall Islands'),
    ('MQ', 'Martinique'),
    ('MR', 'Mauritania'),
    ('MU', 'Mauritius'),
    ('YT', 'Mayotte'),
    ('MX', 'Mexico'),
    ('FM', 'Micronesia'),
    ('MD', 'Moldova'),
    ('MC', 'Monaco'),
    ('MN', 'Mongolia'),
    ('MS', 'Montserrat'),
    ('MA', 'Morocco'),
    ('MZ', 'Mozambique'),
    ('MM', 'Myanmar'),
    ('NA', 'Namibia'),
    ('NR', 'Nauru'),
    ('NP', 'Nepal'),
    ('NL', 'Netherlands'),
    ('AN', 'Netherlands Antilles'),
    ('NC', 'New Caledonia'),
    ('NZ', 'New Zealand'),
    ('NI', 'Nicaragua'),
    ('NE', 'Niger'),
    ('NG', 'Nigeria'),
    ('NU', 'Niue'),
    ('NF', 'Norfolk Island'),
    ('MP', 'Northern Mariana Islands'),
    ('NO', 'Norway'),
    ('OM', 'Oman'),
    ('PK', 'Pakistan'),
    ('PW', 'Palau'),
    ('PA', 'Panama'),
    ('PG', 'Papua New Guinea'),
    ('PY', 'Paraguay'),
    ('PE', 'Peru'),
    ('PH', 'Philippines'),
    ('PN', 'Pitcairn'),
    ('PL', 'Poland'),
    ('PT', 'Portugal'),
    ('PR', 'Puerto Rico'),
    ('QA', 'Qatar'),
    ('RE', 'Reunion'),
    ('RO', 'Romania'),
    ('RU', 'Russian Federation'),
    ('RW', 'Rwanda'),
    ('KN', 'Saint Kitts And Nevis'),
    ('LC', 'Saint Lucia'),
    ('VC', 'St Vincent/Grenadines'),
    ('WS', 'Samoa'),
    ('SM', 'San Marino'),
    ('ST', 'Sao Tome'),
    ('SA', 'Saudi Arabia'),
    ('SN', 'Senegal'),
    ('SC', 'Seychelles'),
    ('SL', 'Sierra Leone'),
    ('SG', 'Singapore'),
    ('SK', 'Slovakia'),
    ('SI', 'Slovenia'),
    ('SB', 'Solomon Islands'),
    ('SO', 'Somalia'),
    ('ZA', 'South Africa'),
    ('ES', 'Spain'),
    ('LK', 'Sri Lanka'),
    ('SH', 'St. Helena'),
    ('PM', 'St.Pierre'),
    ('SD', 'Sudan'),
    ('SR', 'Suriname'),
    ('SZ', 'Swaziland'),
    ('SE', 'Sweden'),
    ('CH', 'Switzerland'),
    ('SY', 'Syrian Arab Republic'),
    ('TW', 'Taiwan'),
    ('TJ', 'Tajikistan'),
    ('TZ', 'Tanzania'),
    ('TH', 'Thailand'),
    ('TG', 'Togo'),
    ('TK', 'Tokelau'),
    ('TO', 'Tonga'),
    ('TT', 'Trinidad And Tobago'),
    ('TN', 'Tunisia'),
    ('TR', 'Turkey'),
    ('TM', 'Turkmenistan'),
    ('TV', 'Tuvalu'),
    ('UG', 'Uganda'),
    ('UA', 'Ukraine'),
    ('AE', 'United Arab Emirates'),
    ('UK', 'United Kingdom'),
    ('UY', 'Uruguay'),
    ('UZ', 'Uzbekistan'),
    ('VU', 'Vanuatu'),
    ('VA', 'Vatican City State'),
    ('VE', 'Venezuela'),
    ('VN', 'Viet Nam'),
    ('VG', 'Virgin Islands (British)'),
    ('VI', 'Virgin Islands (U.S.)'),
    ('EH', 'Western Sahara'),
    ('YE', 'Yemen'),
    ('YU', 'Yugoslavia'),
    ('ZR', 'Zaire'),
    ('ZM', 'Zambia'),
    ('ZW', 'Zimbabwe')]
    flag=1
    words=wt(string)
    #print (words)
    j=len(words)
    for k in range(j):
       for i in v:
        if i[0] in words[k]:
            flag=0
           
            return (i)
            break
    if flag==1:
        for k in range(j):
         for i in v:
          if i[1] in words[k]:
            
           
            return (i)
            break
          
def wt(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences
def extract_zipcode(string):
    words=wt(string)
    #print(words)
    j=len(words)
    l=[]
    s=""
    for k in range(j):
    
      for i in words[k]:
        y=re.compile(r'\d{5}\-\d{4}')
        m= (y.findall(i))
        if m!=[]:
            l.append(m)
    #print (l)
    if l==[]:
       r = re.compile(r'[0-9]+')
       x= (r.findall(string))
    
       #print (x)
   
    
       for k in range(j): 
        for i in words[k]:
           if (i in x and len(i)==5) or(i in x and len(i)==6):
              #print (i)
               s=i
               return (s)
    
    elif l[0]!=[]:
        for i in l[0]:
            for a in i:
            
             if a!='-':
                s+=a
             elif a=='-':
                break
        return(s)
    else:
        return (s)
        
    
        
    
def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    #print (sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    #print(sentences)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON' :
                    names.append(' '.join([c[0] for c in chunk]))
    return names
def name_match(string,desig,names):
    y=string
    y=y.lower()
    desig=desig.lower()
    #print(desig)
    if desig is not None:
        word=wt(desig)
        j=len(word)
        for k in range(j):
            for i in word[k]:
                #print(i)
                if i in y:
                    y=y.replace(i,'')
    fl=[]
    x=[]
    #print(y)
    if y is not None:
       z=wt(y)
       #print(z)
       a=len(z)
    if names!=[]:
        for i in names:
            i=i.lower()
            x.append(i)
    #print (x)   
    if x!=[]:
        for k in range(a):
            for i in z[k]:
                for m in range(len(x)):
                    if i in x[m]:
                        fl.append(i)
        
    #print(fl)
    return(fl)
def extract_organisation(document):
    organisations = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'ORGANIZATION':
                    organisations.append(' '.join([c[0] for c in chunk]))
    return organisations

if __name__ == '__main__':
    ## Input Image
    input = sys.argv[1:]
    #print(input)
    img=cv2.imread(input[0])

    #img = cv2.imread('biz/Img_461.jpg')
    scaling_factor=0.6



##Resize Image
    #r = 1200.0 / img.shape[1]
    #dim = (1200, int(img.shape[0] * r))
    #img = cv2.resize(img, dim, cv2.INTER_AREA)
    img= cv2.resize(img, (0,0),fx=0.5,fy=0.5)

#ratio = img.shape[0] / 300.0

#Save a copy
    img1=img

## Display Image
    #cv2.imshow('original',img)
    #

## Convert to GrayScale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## Adaptive thresholding
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,99,14)

    #cv2.imshow('edges',img)
    #

## Applying Erosion
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 3)
    #cv2.imshow('erosion',erosion)
    #


    im,contours, hierarchy = cv2.findContours(erosion,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)


    contours = sorted(contours, key = cv2.contourArea, reverse = True)
#draw it

    img_copy = img
    img_orig = img1



    mode=0
    c = 0
#for c in range(len(contours)):
    epsilon = 0.05 * cv2.arcLength(contours[c], True)
    approx = cv2.approxPolyDP(contours[c], epsilon, True)
    if len(approx) == 4:
        cv2.drawContours(img_orig, [approx], -1, (255, 0, 0), 2)
        

    #print(len(approx))

    #cv2.imshow('sam',img_orig)
    #

#print(contours[0])


#print(approx)
#print(approx[1])

        app_x = sorted(approx,key=lambda item: (item[0][0]))
        tl = min(app_x[:2],key = lambda item: item[0][1]) 
        tr = max(app_x[:2],key = lambda item: item[0][1])

        app_y = sorted(approx,key=lambda item: (item[0][0]),reverse=True)
        bl= min(app_y[:2],key = lambda item: item[0][1])
        br = max(app_y[:2],key = lambda item: item[0][1])

        approx = (tl,tr,br,bl)


#print(tl)

        pts = np.float32(list(reversed(approx)))

#print(br[1])
#print((pts))
        size = (br[0][1]-tl[0][1]-20,br[0][0]-tl[0][0]-20)
    #print(size)

        transmtx = cv2.getPerspectiveTransform(pts, np.array([[0,0], [size[0],0], size, [0, size[1]]],np.float32))
#print (transmtx)

        quad = cv2.warpPerspective(img1, transmtx, size)
#quad  = cv2.transpose(quad)

        quad = cv2.transpose(quad)
        quad = cv2.flip(quad,1)


    #cv2.imshow('warp',quad)
    #

        cv2.imwrite('roi.png',quad)

        process_image(quad,0)
    else:
        mode=1
        process_image(img1,mode)
