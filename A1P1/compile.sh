pip3 install -r requirements.txt
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords
URL='https://owncloud.iitd.ac.in/owncloud/index.php/s/5bypRrMfaPRaAS5/download'
wget $URL -O pipeline.p