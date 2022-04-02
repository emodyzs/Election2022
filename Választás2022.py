from ezcharts import *
from ezdate import *
from Választás2022_lib import *
import pandas as pd
from pandasql import sqldf

def sqlquery(q): 
    return sqldf(q, globals())

matplot_init()
#tbl=pd.read_csv(r"C:\Users\Zsolt\Downloads\DATA_Választási_közvélemény-kutatások_Magyarországon.xlsx - Adatok.csv",sep=',',parse_dates=True)



def fn_teszt():
    import matplotlib.pyplot as plt
    plt.plot([0,1,2,3,4],[2,3,2,1,0])

    plt.fill_betweenx(plt.ylim(),1,2,color='1.0',alpha=0.3)
    plt.text(1.1,1.3,'sávfelirat',rotation='vertical')

    plt.show()




tbl=pd.read_excel(r"C:\Users\Zsolt\Downloads\DATA_Választási_közvélemény-kutatások_Magyarországon (6).xlsx",sheet_name='Adatok',parse_dates=True)

# Felesleges oszlopok
#tbl=tbl.drop(['UnixStart','UnixEnd','UnixTime'],axis=1)

# ZRI helyett Závecz
tbl.loc[tbl['Adatgazda']=='ZRI','Adatgazda']='Závecz'


# Szöveges oszlopok esetén nan (üres érték) helyett ''  (üres string;  a szűréseket egyszerűsíti)
tbl[['Mód','Fókusz']]=tbl[['Mód','Fókusz']].fillna('')

# Dátum-oszlop:  időszak középdátuma
tbl['Dátum']=tbl['Kezdet'] + (tbl['Vég'] - tbl['Kezdet'])/2

# tbl['Dátum']=tbl['Vég']

# Pártoknál szereplő százalékok összegzése  (a normalizáláshoz kell)
pártok = ['Fidesz','MSZP-P','Jobbik','LMP','DK','Együtt','MM','MMM','Közös lista','MKKP','MH','Egyéb párt']
    # - FIGYELEM:  idővel változhat a felsorolás, ezért ellenőrizni kell az Excel-ben
tbl['PártokSum']=tbl[pártok].sum(axis=1)          # megfelelően kezeli az értékekt nem tartalmazó (NaN) cellákat is

# Ellenzéki pártok százalékainak összegzése
ellenzékipártok =['MSZP-P','Jobbik','LMP','DK','Együtt','MM','MMM','Közös lista']       # A "Közös lista" vagylagos a többivel 
tbl['Ellenzék']=tbl[ellenzékipártok].sum(axis=1)          # jól kezeli az NaN cellákat is

pártok.append('Ellenzék')

# Százalékok normalizálása  (Pártok százalékai adják ki a 100%-t, pl. a teljes népességhez viszonyított százalékok esetén szükséges)
for colname in pártok:
    tbl[colname] = (tbl[colname] / tbl['PártokSum']) * 100


# Felesleges sorok elhagyása:
# - PártokSum==0        96 sor (772-ből)
# - KIIKTATVA   Mód='Online': az online közvéleménykutatások kevésbé megbízhatóak      116 sor (772-ből)
# - részleges vagy problémás közvéleménykutatások  (Fókusz="csak*", "Téves*", "Speciális")
# - csak a rendszeresen publikáló közvéleménykutatók kellenek
adatgazdák = 'Publicus,Závecz,Nézőpont,Századvég,IDEA,Medián,Republikon,Iránytű'.split(',')
#adatgazdák='Závecz,Republikon,IDEA,Publicus,Medián,Nézőpont'.split(',')
where_adatgazdák = ' or '.join(['Adatgazda="' + x + '"' for x in adatgazdák])

pártok_quoted=['"' + x + '"' for x in pártok]

tbl=sqlquery('select "Adatok bázisa",Adatgazda,Dátum,Fókusz,Mód,Minta,' + ','.join(pártok_quoted) + ',PártokSum    from tbl  ' +
    ' where PártokSum>0 and Fókusz not like "csak%" and Fókusz not like "téves%" and Fókusz<>"Speciális"' +
    ' and (' + where_adatgazdák + ')' )


# Sorok összevonása az "Adatgazda"+"Dátum" oszlop ismétlődő értékeire, a számoszlopok átlagolásával (pl "összes megkérdezett" és "pártválasztók")
#   - azonos felmérésekhez tartozó rekordok összevonása; felmérésenként csak egy összevont rekord legyen
#   - az egyéb szöveges oszlopok eltűnnek (pl. Fókusz,Mód  )
tbl2=tbl.groupby(['Adatgazda','Dátum'],as_index=False).mean()



tbl2=tbl2.set_index('Dátum')
tbl2.index=pd.to_datetime(tbl2.index)

tblinfo(tbl2,'info')
tblinfo(tbl2,'0')


annotcircles_fidesz=[
    {'x':'2018.04.18','y':47.36,'caption':'Választás 2018\nFidesz','color':'orange'},        # 47,36
    {'x':'2019.05.26','y':52.56,'caption':'Eu 2019\nFidesz','color':'orange'},               # 52.56 (a VoxPopuli 52-t mutat)
    {'x':'2019.10.13','y':52.8,'caption':'Önkormányzati\nFidesz','color':'orange'},          
                # 54.27  (főpolgármester és megyei listák) A Vox Populi valamilyen korrekciót alkalmazhatott (?)
]

annotcircles_ellenzék=[
    {'x':'2018.04.18','y':48.22,'caption':'Választás 2018\nEllenzék','color':'blue'},          # 48,22
    {'x':'2019.05.26','y':41.11,'caption':'Eu 2019\nEllenzék','color':'blue'},                 # 41.11
    {'x':'2019.10.13','y':41.04,'caption':'Önkormányzati\nEllenzék','color':'blue'},           # 41.04
                # főpolgármester és megyei listák. Itt nem látszik eltérés a Vox Populi-tól
]

annotcircles = annotcircles_fidesz+annotcircles_ellenzék


annotbands=[
    # {'koord':'2018-04-08','koord2':'2018-04-18','caption':'Országgyűlési választás, 2018'},
    # {'koord':'2019-05-16','koord2':'2019-05-26','caption':'Európai parlamenti választás, 2019'},
    # {'koord':'2019-10-03','koord2':'2019-10-13','caption':'Önkormányzati választás, 2019'},

    {'koord':'2020-03-15','koord2':'2020-05-31','caption':'1. hullám','fontsize':7},
    {'koord':'2020-09-01','koord2':'2021-01-10','caption':'2. hullám','fontsize':7},
    {'koord':'2021-02-15','koord2':'2021-04-30','caption':'3. hullám','fontsize':7},
    {'koord':'2021-09-16','koord2':'2021-10-16','caption':'Előválasztás','fontsize':7},
    # {'koord':'2021-11-01','koord2':'2022-01-15','caption':'4. hullám','fontsize':7},
    {'koord':'2022-02-24','koord2':fn_today('-'),'caption':'Orosz invázió','fontsize':7}
    ]


# Gauss mozgóátlag az összes mérési pontra
tbl3=tbl2.copy()
tbl3=tbl3.sort_index()

tblinfo(tbl3,'plot',cols='ellenzék,fidesz',area=False,
        plttype='scatter gauss',gausswidth=80,resample=400,normalize=False,extend_to='2022.04.03',
        width=0.85,height=0.65,
        suptitle='Fidesz és ellenzék, összes mérés 2018-tól',
        left=0.07,right=0.9,bottom=0.11,
        y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
        xlabel='Dátum',xtickerstep='date',ynumformat='0f%',
        annotate='max min last',annotatecaption='{label} ({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles)    

# tblinfo(tbl3,'plot',cols='ellenzék,fidesz',query='Dátum<="2018-04-08"',area=False,
#         plttype='scatter gauss',gausswidth=100,resample=200,normalize=False,
#         width=0.9,height=0.7,
#         suptitle='Fidesz és ellenzék, a 2018-as választásig',
#         left=0.07,right=0.86,bottom=0.11,
#         y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
#         xlabel='Dátum',xtickerstep='date',ynumformat='0f%',
#         annotate='max last',annotatecaption='{label} ({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles)    



# EREDŐ GÖRBÉK SZÁMÍTÁSA
# Kétféle átlagot számol:
#  Average:  az eredő eltérés minimalizálása a közvéleménykutatók milyen súlyozással a legkisebb
#  Average2:  közvéleménykutatónaként az eltérés-átlag reciproka a súlyfaktor 

# adatgazdák='Závecz,Republikon,IDEA,Publicus,Medián,Nézőpont'
adatgazdák='Závecz,Republikon,IDEA,Publicus,Medián,Nézőpont'
pártok=['Fidesz','Ellenzék']
tbl2 = fn_inweights(tbl2.copy(),adatgazdák,pártok,annotcircles_fidesz,annotcircles_ellenzék,'2018.01.01','2022.04.03')



#tblinfo(tbl2,'browse',cols='dátum,adatgazdál,fidesz,közös lista,dk',groupby='Adatgazda')



diagrams=[
    # ['Eltolással','Eltolással'],
    ['Average2','A közvéleménykutatások súlyozott átlaga'],
    # ['Average','A közvéleménykutatások súlyozott átlaga (legkisebb eltérésre optimalizált súlyfaktorok)'],
    ]
for adatgazda,title in diagrams:
    tblinfo(tbl2,'plot',cols='ellenzék,fidesz',groupby='Adatgazda:' + adatgazda,area=False,
            resample=50,gausswidth=30,plttype='gauss',normalize=False,extend_from='2018.01.01',extend_to='2022.04.03',
            width=0.6,height=0.8,
            suptitle='Fidesz és ellenzék, 2018-tól',
            title=title,
            left=0.07,right=0.86,bottom=0.11,
            y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
            xlabel='Dátum',xtickerstep='date',ynumformat='0f%',
            annotate='max min last',annotatecaption='{label} ({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles)    




# adatgazdák='Závecz,Publicus,Republikon,Iránytű,IDEA,Medián,Nézőpont,Századvég'
adatgazdák='Závecz,Republikon,Publicus,IDEA,Medián,Nézőpont'
tblinfo(tbl2,'plot',cols='fidesz',groupby='Adatgazda:' + adatgazdák,area=False,
        resample=100,gausswidth=15,plttype='gauss',normalize=False,extend_from='2018.01.01',extend_to='2022.04.03',
        suptitle='Fidesz, 2018-tól',
        width=0.6,height=0.8,left=0.07,right=0.86,bottom=0.11,
        y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
        xtickerstep='date',ynumformat='0f%',
        annotate='localmax1 last',annotatecaption='{label} ({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles_fidesz)    


# Publicus,Závecz,Nézőpont,IDEA,Medián,Republikon,Századvég,Iránytű
tblinfo(tbl2,'plot',cols='ellenzék',groupby='Adatgazda:' + adatgazdák,
        area=False,resample=100,gausswidth=15,plttype='gauss',normalize=False,
        extend_from='2018.01.01',extend_to='2022.04.03',
        suptitle='Ellenzék, 2018-tól',
        width=0.6,height=0.8,left=0.07,right=0.86,bottom=0.11,
        y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
        xtickerstep='date',ynumformat='0f%',
        annotate='localmax1 last',annotatecaption='{label} ({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles_ellenzék)    


# Egy-egy közvéleménykutató részletes ábrái (fidesz-ellenzék)
annotbands=[
    {'koord':'2018-04-08','koord2':'2018-04-18','caption':'Választás','fontsize':6},
    {'koord':'2019-05-16','koord2':'2019-05-26','caption':'EU parl.','fontsize':6},
    {'koord':'2019-10-03','koord2':'2019-10-13','caption':'Önkorm','fontsize':6},
    {'koord':'2020-03-15','koord2':'2020-05-31','caption':'1. hullám','fontsize':6},
    {'koord':'2021-02-15','koord2':'2021-04-30','caption':'3. hullám','fontsize':6},
    {'koord':'2021-09-16','koord2':'2021-10-16','caption':'Előválasztás','fontsize':6},
    {'koord':'2022-02-24','koord2':'end','caption':'Invázió','fontsize':6}
    ]

# for i in range(len(annotbands)): del annotbands[i]['caption']
for i in range(len(annotcircles)): del annotcircles[i]['caption']

# adatgazdák='Závecz//Publicus//Republikon//Iránytű//IDEA//Medián//Nézőpont//Századvég'
adatgazdák='Závecz//Publicus//Medián//Republikon//IDEA//Nézőpont'
tblinfo(tbl2,'plot',cols='ellenzék,fidesz',groupby='Adatgazda:' + adatgazdák,area=False,
        resample=100,gausswidth=15,plttype='scatter gauss',
        extend_from='2018.01.01',extend_to='2022.04.03',
        suptitle='Közvéleménykutatók mérései 2018-tól',
        left=0.05,right=0.95,bottom=0.1,
        normalize=False,y1=30,y2=60,x1="2018-01-01",x2='2022-04-03',
        xlabel='aaa',xtickerstep='date',tickpad=1,ticksize=7,ynumformat='0f%',
        annotate='last',annotatecaption='{label}\n({y:.1f}%)',annotbands=annotbands,annotcircles=annotcircles)




