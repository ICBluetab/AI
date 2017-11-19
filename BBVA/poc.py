# -*- coding: utf-8 -*-
import numpy as pd
import pandas as pd

xl = pd.ExcelFile("../LargeFiles/bbva/Parque_POC_BluePad_Oct 2017_F.xlsx")

print "sheet names ",  xl.sheet_names

#df_info = xl.parse(u'Info cajero')
#df_disp = xl.parse(u'Disponibilidad')
#df_oper = xl.parse(u'Operaciones')
#df_hist = xl.parse(u'Histórico dispensa')
df_aver = xl.parse(u'Averias')
#df_aler = xl.parse(u'Alertas')

#print "Info cajero"
#print df_info.info()

#print "Disponibilidad"
#print df_disp.info()

#print "Operaciones"
#print df_oper.info()

#print "Histórico dispensa"
#print df_hist.info()

print "Averias"
print df_aver.info()

#print df_aler.info()
#print "Alertas"

df = df_aver[df_aver.XTI_IMPUT == 'Y']
df = df[['CAJERO', 'TIM_APERTURA', 'QNU_AVE30']]
df.columns = ['cajero', 'fecha', 'num averias']
df.to_datetime(df['fecha'], infer_datetime_format=True)
df['target'] = 1

df.to_csv("coocked.csv", index=False)
