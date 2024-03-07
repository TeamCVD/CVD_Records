import numpy as np
import pandas as pd
import os

# from google.colab import drive
# drive.mount('/content/drive')
# os.listdir("/content/drive/My Drive/LSEG_Dataset")

# Read the available Dataset
data = pd.read_csv("/content/drive/My Drive/LSEG_Dataset/marked_sard_vdisc_train.csv")

# Extract data with only one vulnerability and concatenate it with the entries without vulnerabilities

cwe119_vulnerable = data[(data['CWE-119']==True)&(data['CWE-120']==False)&(data['CWE-469']==False) & (data['CWE-476']==False) & (data['CWE-OTHERS']==False)]
cwe120_vulnerable = data[(data['CWE-119']==False)&(data['CWE-120']==True)&(data['CWE-469']==False) & (data['CWE-476']==False) & (data['CWE-OTHERS']==False)]
cwe469_vulnerable = data[(data['CWE-119']==False)&(data['CWE-120']==False)&(data['CWE-469']==True) & (data['CWE-476']==False) & (data['CWE-OTHERS']==False)]
cwe476_vulnerable = data[(data['CWE-119']==False)&(data['CWE-120']==False)&(data['CWE-469']==False) & (data['CWE-476']==True) & (data['CWE-OTHERS']==False)]
cweOTH_vulnerable = data[(data['CWE-119']==False)&(data['CWE-120']==False)&(data['CWE-469']==False) & (data['CWE-476']==False) & (data['CWE-OTHERS']==True)]
non_vulnerable = data[(data['CWE-119']==False)&(data['CWE-120']==False)&(data['CWE-469']==False) & (data['CWE-476']==False) & (data['CWE-OTHERS']==False)]

#Concatenate the vulnerable and non-vulnerable data
cwe119_Data = pd.concat([cwe119_vulnerable,non_vulnerable])
cwe120_Data = pd.concat([cwe120_vulnerable,non_vulnerable])
cwe469_Data = pd.concat([cwe469_vulnerable,non_vulnerable])
cwe476_Data = pd.concat([cwe476_vulnerable,non_vulnerable])
cweOTH_Data = pd.concat([cweOTH_vulnerable,non_vulnerable])

# Extract the necessary columns
cwe119_Data = cwe119_Data.loc[:,['code','CWE-119','Label']]
cwe120_Data = cwe120_Data.loc[:,['code','CWE-120','Label']]
cwe469_Data = cwe469_Data.loc[:,['code','CWE-469','Label']]
cwe476_Data = cwe476_Data.loc[:,['code','CWE-476','Label']]
cweOTH_Data = cweOTH_Data.loc[:,['code','CWE-OTHERS','Label']]

# Specify the path variables
cwe119_location = "/content/drive/My Drive/LSEG_Dataset/cwe119_Data.csv"
cwe120_location = "/content/drive/My Drive/LSEG_Dataset/cwe120_Data.csv"
cwe469_location = "/content/drive/My Drive/LSEG_Dataset/cwe469_Data.csv"
cwe476_location = "/content/drive/My Drive/LSEG_Dataset/cwe476_Data.csv"
cweOTH_location = "/content/drive/My Drive/LSEG_Dataset/cweOTH_Data.csv"

# convert as .csv files and save
cwe119_Data.to_csv(cwe119_location,index=False)
cwe120_Data.to_csv(cwe120_location,index=False)
cwe469_Data.to_csv(cwe469_location,index=False)
cwe476_Data.to_csv(cwe476_location,index=False)
cweOTH_Data.to_csv(cweOTH_location,index=False)
