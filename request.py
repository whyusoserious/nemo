from urllib.request import urlopen
import os
from bs4 import BeautifulSoup
import pandas as pd

# Polar NMs....
station_list = sorted(["ROME"])
# Note ! Kiel, DRBS, MOSC -->> unstable NMs

Begin_datetime, End_datetime = "2021-01-01", "2022-12-31"
start_year, start_month, start_day = str(Begin_datetime).split("-")
end_year, end_month, end_day = str(End_datetime).split("-")
os.system("mkdir -p " + str(start_year) + "-" + str(end_year) + "/Measured_data")
for NM_Station_ in station_list:
    print("\n Downloading data for ---> ", NM_Station_)
    try:
        url_ = f"http://nest.nmdb.eu/draw_graph.php?formchk=1&stations[]={NM_Station_}&output=ascii&tabchoice=ori" \
               f"&dtype=corr_for_efficiency&date_choice=bydate&start_year={start_year}&start" \
               f"_month={start_month}&start_day={start_day}&start_hour=00&start_min=00&end_year=" \
               f"{end_year}&end_month={end_month}&end_day={end_day}&end_hour=23&end_min=59&yunits=0"
        URL = url_.format(NM_Station_=NM_Station_, start_day=start_day, start_month=start_month, start_year=start_year,
                          end_day=end_day, end_month=end_month, end_year=end_year)
        WebR = urlopen(URL)
        parserHTML = WebR.read()
        Bsoup = BeautifulSoup(parserHTML, features="html.parser")
        Tx = Bsoup.find_all('pre')[0].text
        Tx = Tx.replace("start_date_time   1HCOR_E", "# Datetime, Counts/s")
        File_ = open(str(start_year) + "-" + str(end_year) + "/Measured_data/" + str(NM_Station_) + ".txt", "w")
        for line in Tx.split('\n'):
            if line.startswith('20') or line.startswith('19'):
                File_.write(line + '\n')
        File_.close()
    except:
        print(NM_Station_, " ! Could not download and process data...")
print(' \n\n Done saving datasets !!!!!!.......\n\n')
