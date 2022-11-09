import requests # to get image from the web
import shutil # to save it locally
import os
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
import time
from itertools import repeat

DATA_PATH = '/Users/travistang/Documents/TorchScene/'

numberImages = 0 

def download_image(url, merchant_id, folder_name):
    
    global numberImages 
    
    ## Set up the image URL and filename
    image_url = url

    new_path = f"{DATA_PATH}/{folder_name}/"
    file_path = new_path + merchant_id + '.jpeg'

    if os.path.exists(file_path):
        # print("Exists: ", file_name, '\n')
        return

    try:
        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True

            if not os.path.exists(new_path):
                os.makedirs(new_path)

            # Open a local file with wb ( write binary ) permission.
            with open(file_path,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                numberImages += 1

            #sleep for half second to prevent getting bumped out
            time.sleep(0.5)

            #  print('Downloaded: ',file_name, '\n')
        else:
              print('Image Couldn\'t be retreived')
                
    except Exception as e:
        print('Error Type: {0}\n'.format(e.__class__.__name__))
        print('Error Message: {0}\n'.format(e))
        
# delete images that do not exist
def test_and_delete(path):
    try:
        image = Image.open(path)
    
    except:
        os.remove(path)  
        print(f'\n{path} not valid and deleted.\n')
        

if __name__ == '__main__':
    img_df = pd.read_csv('/Users/travistang/Documents/TorchScene/data/csv/Travis Fake_Merchant_Tagging_2022 - Combined Set.csv')
    img_df = img_df.dropna(subset=['outlet_photo_url'])

    numberImages = 0
    n_process = 4
    pool = Pool(n_process)
    start = datetime.now()

    url_sel_list = img_df['outlet_photo_url']
    merchant_sel_list = img_df['saudagar_id']
    folder_name = "data/images/victor_set"

    pool.starmap(download_image, zip(url_sel_list, merchant_sel_list, repeat(folder_name)))
    duration = datetime.now() - start
    print(f'Trend calculation for {len(merchant_sel_list)} outlets completed in {duration}')

    pool.close()
    pool.join()