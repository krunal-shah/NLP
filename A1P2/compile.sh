#!/bin/bash
# pip3 install -r requirements.txt
# python3 -m nltk.downloader stopwords
URL_model='https://owncloud.iitd.ac.in/owncloud/index.php/s/n23zgcwESq8Ciws/download'
URL_dict='https://owncloud.iitd.ac.in/owncloud/index.php/s/3wYFgiTE77gxpYf/download'
wget $URL_model -O model.pt
wget $URL_dict -O dict.pickle
