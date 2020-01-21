import os
import time
import requests
import sys




def retrieve_html():
    for year in range(2013,2019):
        for month in range(1,13):
            if(month<10):
                url='http://en.tutiempo.net/climate/0{}-{}/ws-421820.html'.format(month, year)
                # 0{} for month and {} for year indicates, month and year for air quality
                # insted of %i we can use empty colones like 0%i-%i one for month and another for year
            else:
                url='http://en.tutiempo.net/climate/{}-{}/ws-421820.html'.format(month, year)
            
            # after the else block retrive the text by using request lib. with this get all the values from HTML
            texts=requests.get(url)
            #when we retrive the url we have do UTF ENCODING
            text_utf=texts.text.encode('utf=8')
            
            # /data mens save the data in that data folder and in side html folder save that
            if not os.path.exists("Data/Html_Data/{}".format(year)):
                os.makedirs("Data/Html_Data/{}".format(year))
            with open("Data/Html_Data/{}/{}.html".format(year,month),"wb") as output:       #wb = write bite mode
                output.write(text_utf)
            
        sys.stdout.flush()


if __name__=="__main__":

    start_time=time.time()
    retrieve_html()
    stop_time=time.time()
    print("Time taken {}".format(stop_time-start_time))
