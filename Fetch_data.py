import os
import wget
import netCDF4
import matplotlib.pyplot as plt
import numpy as np

def FloatToString(d):

    if d >= 10:
        return str(d)
    else:
        return "0" + str(d)

def Fetch_data(type):
    fn = os.path.join(os.path.join(os.getcwd(), 'ARGO'), type)
    for year in range(2018, 2019):
        for month in range(7, 13):
            print(year, month)
            for day in range(1, 32):
                try:
                    url = "https://data-argo.ifremer.fr/geo/" + type + "/"
                    url += FloatToString(year) + "/"
                    url += FloatToString(month) + "/"
                    url += FloatToString(year) + FloatToString(month) + FloatToString(day) + "_prof.nc"
                    wget.download(url, fn)
                except:
                    print("Can not open the file.")

Fetch_data("indian_ocean")

data = netCDF4.Dataset(os.path.join(os.path.join(os.getcwd(), "ARGO\\atlantic_ocean"), "20200101_prof.nc"), format="NETCDF4")
a = data["PRES"]
a = data["JULD"]
DATA_TYPE
FORMAT_VERSION
HANDBOOK_VERSION
REFERENCE_DATE_TIME
DATE_CREATION
DATE_UPDATE
PLATFORM_NUMBER
PROJECT_NAME
PI_NAME
STATION_PARAMETERS
CYCLE_NUMBER
DIRECTION
DATA_CENTRE
DC_REFERENCE
DATA_STATE_INDICATOR
DATA_MODE
PLATFORRM_TYPE
FLOAT_SERIAL_NO
FIRMWARE_VERSION
WMO_INST_TYPE
JULD
JULD_QC
JULD_LOCATION
LATITUDE
LONGITUDE
POSITION_QC
POSITIONING_SYSTEM
PROFILE_PRES_QC
PROFILE_TEMP_QC
VERTICAL_SAMPLING_SCHEME
CONFIG_MISSION_NUMBER
PRES
PRES_QC
PRES_ADJUSTED
PRES_ADJUSTED_QC
PRES_ADJUSTED_ERROR
TEMP
TEMP_QC
TEMP_ADJUSTED
TEMP_ADJUSTED_QC
TEMP_ADJUSTED_ERROR
PARAMETER
SCIENTIFIC_CALIB_EQUATION
SCIENTIFIC_CALIB_COEFFICIENT
SCIENTIFIC_CALIB_COMMENT
SCIENTIFIC_CALIB_DATE
HISTORY_INSTITUTION
HISTORY_STEP
HISTORY_SOFTWARE
HISTORY_SOFTWARE_RELEASE
HISTORY_REFERENCE
HISTORY_DATE
HISTORY_ACTION
HISTORY_PARAMETER
HISTORY_START_PRES
HISTORY_STOP_PRES
HISTORY_PREVIOUS_VALUE
HISTORY_QCTEST

def compactor(pres_target, year):

    #pos is: time,lat,long
    time = np.array([])
    lat = np.array([])
    long = np.array([])
    temp = np.array([])
    pres = np.array([])
    oceans = ["atlantic_ocean", "pacific_ocean", "indian_ocean"]

    for month in range(1, 13):
        for day in range(1, 32):
            for type in oceans:
                domain = os.path.join(os.path.join(os.getcwd(), "ARGO"), type)
                filename = os.path.join(domain, str(year)+FloatToString(month)+FloatToString(day)+"_prof.nc")
                try:
                    data = netCDF4.Dataset(filename, format="NETCDF4")

                    DATA_MODE = np.array(data["DATA_MODE"])
                    Irow = (DATA_MODE == b"D")

                    PRES, PRES_QC = np.array(data["PRES"])[Irow, :], np.array(data["PRES_QC"][Irow, :])
                    PRES_ADJUSTED = np.array(data["PRES_ADJUSTED"])[Irow, :]
                    PRES_ADJUSTED_QC = np.array(data["PRES_ADJUSTED_QC"][Irow, :])

                    Icol = (PRES_ADJUSTED == pres_target)
                    Icols = np.array(np.sum(Icol, axis=1), dtype=bool)

                    JULD, JULD_QC = np.array(data["JULD"][Irow][Icols]), np.array(data["JULD_QC"][Irow][Icols])

                    LATITUDE, LONGITUDE = np.array(data["LATITUDE"][Irow][Icols]), np.array(data["LONGITUDE"][Irow][Icols])
                    POSITION_QC = np.array(data["POSITION_QC"][Irow][Icols])

                    TEMP  = np.array(data["TEMP"][Irow, :][Icol])
                    TEMP_QC = np.array(data["TEMP_QC"][Irow, :][Icol])
                    TEMP_ADJUSTED = np.array(data["TEMP_ADJUSTED"][Irow, :][Icol])
                    TEMP_ADJUSTED_QC = np.array(data["TEMP_ADJUSTED_QC"][Irow, :][Icol])

                    PRES, PRES_QC, PRES_ADJUSTED = PRES[Icol], PRES_QC[Icol], PRES_ADJUSTED[Icol]
                    PRES_ADJUSTED_QC = PRES_ADJUSTED_QC[Icol]

                    QC = ((JULD_QC == b"1") | (JULD_QC == b"2") | (JULD_QC == b"5") | (JULD_QC == b"8")) & \
                         ((POSITION_QC == b"1") | (POSITION_QC == b"2") | (POSITION_QC == b"5") | (POSITION_QC == b"8")) & \
                         ((TEMP_ADJUSTED_QC == b"1") | (TEMP_ADJUSTED_QC == b"2") | \
                          (TEMP_ADJUSTED_QC == b"5") | (TEMP_ADJUSTED_QC == b"8")) & \
                         ((PRES_ADJUSTED_QC == b"1") | (PRES_ADJUSTED_QC == b"2") | \
                          (PRES_ADJUSTED_QC == b"5") | (PRES_ADJUSTED_QC == b"8"))

                    time = np.hstack((time, JULD[QC]))
                    lat = np.hstack((lat, LATITUDE[QC]))
                    long = np.hstack((long, LONGITUDE[QC]))
                    temp = np.hstack((temp, TEMP_ADJUSTED[QC]))
                    pres = np.hstack((pres, PRES_ADJUSTED[QC]))

                except:
                    print("Can not open the file:", filename)

    file_name = os.path.join(os.path.join(os.getcwd(), "ARGO"), "compact10")
    np.save(os.path.join(file_name, str(year)), np.vstack((time, lat, long, temp, pres)))


for i in range(2000, 2022):

    compactor(10, i)




import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':20})
h = np.linspace(0, 4 * np.pi, 1000)
func = lambda p, x: 1 - np.cos(h) / (1 + p + p * np.sin(h))
plt.plot(h, func(1, h), label="p="+str(1))
plt.plot(h, func(5, h), label="p="+str(5))
plt.plot(h, func(10, h), label="p="+str(10))
plt.plot(h, func(0.5, h), label="p="+str(0.5))
plt.plot(h, func(50, h), label="p="+str(50))
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

func = lambda g, x: 1 - (g * (1 + np.sin(h)) - np.cos(h) + 1) / ((1 + g * (1 + np.sin(h))) ** 2)
plt.plot(h, func(1, h), label="p="+str(1))
plt.plot(h, func(5, h), label="p="+str(5))
plt.plot(h, func(10, h), label="p="+str(10))
plt.plot(h, func(0.5, h), label="p="+str(0.5))
plt.plot(h, func(50, h), label="p="+str(50))
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

