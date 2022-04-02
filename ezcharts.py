# from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb
import numpy as np
import pandas as pd
from datetime import datetime,date,timedelta
import copy
# import ez
import colorsys
import math
import re
from scipy.interpolate import interp1d,UnivariateSpline
import statsmodels.tsa.stattools as stat
from sklearn import linear_model

from ezhelper import *

import config
import locale


# Matplotlib háttrébeállítások
def matplot_init():
    ''' FIGYELEM  a debug megfigyelt változói közé ne vedd fel a plt.gcf() vagy a plt.gca() hívást,
      mert a debug ilyen helyből meg fog jelenni egy diagram ablak
    '''

    locale.setlocale(locale.LC_TIME, "hungarian")

    mpl.style.use('seaborn')
    plt.rcParams.update({'figure.figsize': (10, 5), 'figure.dpi': 100, 'figure.titlesize':'x-large', 'figure.subplot.hspace':0.25,
                         'lines.linewidth':1, 'lines.markersize':3, 'axes.prop_cycle':plt.cycler("color", plt.cm.tab10.colors),
                         'axes.grid':True, 'axes.labelsize':'small', 'xtick.labelsize':'small', 'ytick.labelsize':'small',
                         'font.sans-serif':'Century Gothic', 'axes.formatter.use_locale':True})

    # plt.ioff()      # interactive mode kikapcsolása


    # plt.rcParams.keys()     # lehetséges paraméterek listája


    pd.set_option('display.max_columns',14)         # efelett truncated nézet (default 6;  adatforrás felrétképezésekeor tovább növelhető)
    pd.set_option('display.max_colwidth',50)        # oszloponként max milyen szélesség (az oszlopnevek láthatósága miatt nem lehet túl kicsi, default 80)
    pd.set_option('display.width',400)              # ha ennél nagyobb hely kell, akkor blokkszintű tördelés
    pd.set_option('display.max_rows',200)           # default 60   Efölött truncated nézet (head és tail esetén is)
    pd.set_option('display.min_rows',14)            # truncated nézetben hány sor jelenjen meg (default 10)
    pd.set_option('mode.use_inf_as_na',True)        # a végtelent és az üres stringeket is kezelje NA-ként (isna)
    pd.set_option('display.float_format', '{:,.10g}'.format)    # 10 számjegy, ezres határolással (sajnos csak ',' lehet)



def FvTeszt():
    return 'hahó9'

# plot műveletek tesztelése
def fn_tesztplot():
    plt.plot([0,1,2,3,4],[2,3,2,1,0])

    plt.fill_betweenx(plt.ylim(),1,2,color='1.0',alpha=0.3)
    plt.text(0.3,1,'sávfelirat',rotation='vertical')

    plt.show()


def fn_readcsv(path,sep=',',encoding='utf-8'):
    ''' 
    path:  r-string kell (forrás: path másolás a fájlkezelőben)
    sep:  pontosvessző is lehet (gyakori)
    encoding:  default "utf-8", "cp1250",  wheathercloud: "utf-16-le"
       ha nem jók már a mezőnevek sem, akkor nézd meg commander-rel (az utf-16 könnyen felismerhető, 00 karakterek minden második helyen)
    '''
    tblIn=pd.read_csv(path,sep=sep)
    return tblIn






def color_darken(color, amount=0.5):
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])






def SerFromRecords(aRec:list,xname:str='id',yname:str='',sortindex=False):      # -> pd.Series
    # a visszaadott series indexe nem feltétlenül rendezett (ser.sort_index()-re lehet szükség). Az sem elvárás, hogy unique legyen
    aX,aY=unzip(aRec)
    return SerFromArrays(aX,aY,xname,yname,sortindex)

def SerFromArrays(aX:list,aY:list,xname:str='id',yname:str='',sortindex=False):      # -> pd.Series
    # a visszaadott series indexe nem feltétlenül rendezett (ser.sort_index()-re lehet szükség). Az sem elvárás, hogy unique legyen
    # xname: 'date', 'datum' esetén  to_datetime
    ser=pd.Series(aY,aX)
    if xname: ser.index.name=xname
    if yname: ser.name=yname
    if xname in ['date','datum']: ser.index=pd.to_datetime(ser.index)
    if sortindex: ser.sort_index(inplace=True)
    return ser

def TblFromRecords(aRec:list,colnames:str,indexcols:str='',sortindex=False):    
    # colnames:  pl 'id,adat1,adat2'    vesszős felsorolás
    # indexcols:  pl. 'id'        nem kötelező, hogy legyen index (ha nincs, akkor autosorszámozott)
    #    - nem kell unique-nak lennie (ellenőrzés: index.is_unique()), és a rendezettség sem elvárás (plot előtt tbl.sort_index()-re lehet szükség)
    columns=colnames.split(',')
    if indexcols: index=indexcols.split(',') 
    else: index=None
    tbl=pd.DataFrame.from_records(aRec,columns=columns,index=index)
    if sortindex: tbl.sort_index(inplace=True)
    return tbl


def serisempty(ser):
    ''' True, ha None, nincs sora, vagy minden értéke üres (None, nan, '')
    - a ser.empty nem ugyanezt jelenti, mert egy nan értékeket tartalmazó ser-t nem tekint empty-nek
    - az üres string csak akkor minősül üres értéknek, ha be van állítva:  pd.set_option('mode.use_inf_as_na',True)
    '''
    return ser is None or len(ser)==0 or ser.notnull().sum()==0



def servalues(ser,bIndex=False):
    # Értékeloszlás (legyakoribb értékek legelöl)
    if bIndex: return ser.index.value_counts()
    else: return ser.value_counts()
        
def servalue(ser,x,assume_sorted=False):
    # Először megnézi, hogy van-e ilyen x érték a ser x-tömbjében. Ha van, akkor a hozzá tartozó ser-value-t adja vissza.
    # Ha nincs, akkor lineáris interpolációval kéri be az értéket (az x-nek a végpontokon belül kell lennie) 
    # - a végpontokon kívüli tartományokban megbízhatatlan lehet (előtte egy simításra lehet szükség, pl. gauss, spline)
    # - lényegesen időigényesebb  (sok pont esetén lásd servalueA)
    #     Megjegyzés: egy sokkal gyorsabb algoritmust is lehetne írni, a két szomszédos (xy) pár alapján

    try:
        y=ser[x]
        if type(y)==float: return y     # előfordulhat, hogy series-t ad vissza
    except:
        pass

    aXSer=datefloatA(ser.index.array) 
    aYSer=ser.array
    f=interp1d(aXSer,aYSer,assume_sorted)
    aY=f([datefloat(x)])
    return aY[0]

def servalueA(ser,aX,outtype='array'):
    # A ser folytonos függvényként való kiterjesztése (lineáris interpoláció a pontok között, és a végpontokon kívül is) 
    #     és a megadott x-értékekhez a függvényérték bekérése
    # A végpontokon kívüli tartományokban megbízhatatlan lehet (előtte egy simításra lehet szükség, pl. gauss, spline)
    # outtype:  'array' vagy 'dict' vagy 'records'

    aXSer=datefloatA(ser.index.array) 
    aYSer=ser.array

    f=interp1d(aXSer,aYSer,assume_sorted=False)
    if outtype=='array': return f(datefloatA(aX))
    elif outtype=='dict': return dict(zip(aX,f(datefloatA(aX))))
    elif outtype=='records': return zip(aX,f(datefloatA(aX)))


def serfirstvalue(ser):         # az első olyan xy pár, amiben az y nem NaN.  Ha nincs ilyen xy, akkor None,None
    values=ser.values
    ifirst=None
    for i in range(len(values)):
        if pd.isna(values[i]): continue
        ifirst=i
        break
    if ifirst==None: return (None,None)
    else: return (ser.index[ifirst],values[ifirst])

def serlastvalue(ser,maxstep=None):         # az utolsó olyan xy pár, amiben az y nem NaN.  Ha nincs ilyen xy, akkor None,None
    # Elsősorban idősorok esetén alkalmazható
    # Ha nincs utolsó érték, akkor return None
    # maxstep:  max hány lépést (pl. napot) mehet vissza az idősor végétől
    
    values=ser.values

    nlen=len(values)
    rangelast=-1
    if maxstep: 
        rangelast=nlen-maxstep-1
        if maxstep<-1: maxstep=-1
    
    ilast=None
    for i in range(nlen-1,rangelast,-1):
        if pd.isna(values[i]): continue
        ilast=i
        break
    if ilast==None: return (None,None)
    else: return (ser.index[ilast],values[ilast])

def serlocalmax(ser,maxmin='max',halfwidth=2,mindistance=0,sidecut=0):    #  out:  aXy - lokális max-pontok (xy),  legnagyobb y legelöl
    '''
    A visszadatt tömb üres is lehet  (ha nincs egyetlen lokális maxpont sem)
    ser:  az x értékek float vagy datetime típusúak és növekvő sorrendben rendezettek, az y float  (nan is lehet benne)
        - ha a ser nem rendezett x-re, akkor előtte ser=ser.sort_index()
    maxmin:  'max'  vagy 'min'    lokális max-helyek vagy lokális min-helyek
    halfwidth:  
        - a lokális max-helyek elvárt (fél)szélessége   
        - lokális maximum, ha mindkét irányban legalább ennyi szomszédos pontra teljesül a max-elvárás (1-nél nem lehet kisebb)
    mindistance (0-1):  a túl sűrűn következő maxhelyek elhagyása; legalább ekkora távolságnak kell lennie két localmax között (a teljes x-szélességhez viszonyítva)
    sidecut (0-0.5):  a jobb és a bal szél figyelmen kívül hagyása  (a teljes x-szélesség ekkora hányadát kell figyelmen kívül hagyni mindkét szélen)
    '''

    if serisempty(ser): return []
    ser=ser.dropna()

    width=datefloat(ser.index.max()) - datefloat(ser.index.min())    # x-re elvárás a rendezettség, ezért ser.index[-1] és ser.index[0] is lehetne
    if width==0:  return []


    if sidecut>0:
        if sidecut>=0.5: return []
        aX=datefloatA(ser.index)
        indexfirst=len(aX)
        for i in range(len(aX)):
            if aX[i]-aX[0]>sidecut*width:
                indexfirst=i
                break
        indexlast=0
        for i in range(len(aX)-1,-1,-1):
            if aX[-1]-aX[i]>sidecut*width:
                indexlast=i
                break
        if indexlast<indexfirst: return []
        ser=ser[indexfirst:indexlast]

    aL=ser.values
    if halfwidth==None: halfwidth=2
    if halfwidth<1: return []
    


    aIndex=[]   # lokális szélsőértékek pozíciója (max vagy min)
    # Minden pontra vizsgálandó, hogy lokális max-ról van-e szó
    if maxmin=='max':
        for i in range(halfwidth,len(aL)-halfwidth):        
            bOk=True
            for j in range(1,halfwidth+1):
                if aL[i]<=aL[i-j] or aL[i]<=aL[i+j]: 
                    bOk=False
                    break
            if bOk: aIndex.append(i)
    else:   # min
        for i in range(halfwidth,len(aL)-halfwidth):        
            bOk=True
            for j in range(1,halfwidth+1):
                if aL[i]>=aL[i-j] or aL[i]>=aL[i+j]: 
                    bOk=False
                    break
            if bOk: aIndex.append(i)
    
    aPoints=[]
    for i in range(len(aIndex)):
        aPoints.append((ser.index[aIndex[i]],ser.iloc[aIndex[i]]))
    aPoints.sort(key=lambda x: x[1],reverse= (maxmin=='max'))

    if mindistance>0:
        mindistance=mindistance*width
        aPointsL=[]
        aXL=[]
        for point in aPoints:
            mindistanceL=width
            x=datefloat(point[0])
            for xL in aXL:
                distance=abs(x-xL)
                if distance<mindistanceL: mindistanceL=distance
            if mindistanceL>mindistance:
                aXL.append(x)
                aPointsL.append(point)
        aPoints=aPointsL

    return aPoints


def indexlike(tbl_or_ser,like):
    # A megadott szövegrésszel kezdődő indexű rekordok leválogatása (case insensitive)
    # Dataframe-re és Series-re is működik
    # Csak stringindex-szel rendelkező táblázatokra működik (számindex esetén üres táblázat az eredmény)
    # like: ha * van az elején, akkor bárhol előfordulhat az indexben
    
    if str(tbl_or_ser.index.dtype)!='object': 
        print('Warning: indexlike works properly if index is of type string')
        return tbl_or_ser.iloc[0:0]         # üres táblázatot ad vissza
    if len(like)==0: 
        return tbl_or_ser.iloc[0:0]         # üres táblázatot ad vissza

    if like[-1]=='*': like=like[:-1]    # a végén mindenképpen érvényesül a *
    if len(like)==0: return tbl_or_ser.iloc[0:0] 

    if like[:1]=='*' and len(like)>1:
        return tbl_or_ser[tbl_or_ser.index.str.contains(like[1:],case=False)]
    elif len(like)>0: 
        return tbl_or_ser[tbl_or_ser.index.str.contains('^' + like,case=False)]



def info(object,todo='info',**params):
    ''' dataframe esetén tblinfo(tbl,todo,toexcel,cols,groupby,plotparams) '''

    if 'DataFrame' in str(type(object)): return tblinfo(object,todo,params)
    # egyelőre csak dataframe objektumokra működik  (series objektumokra is kidolgozandó)
    else: return    


def tblinfo(tbl,todo='info',toexcel=False,cols='',query='',groupby='',orderby='',**plotparams):
    """A tábla oszlopainak gyors áttekintése (második körben: értékeloszlás egy-egy oszlopra, oszlop átnevezése, törlése, koorelációk). 
    tbl:  pl. egy csv-ből beolvasott tábla;  az esetleges rekordszűréseket előzetesen kell végrehajtani (pl. .query('location=="Hungary"')
       - nem kötelező, hogy unique legyen az index. A plot-hoz általában szükséges, de általában egy groupby oszloppal biztosítható az unicitás.
    todo: 
      "info": (default)  kimutatás az oszlopokról  (colindex, colname, dtype, not_null, unique, distinct_values, most_frequent) 
      "minmax":           colindex, colname, dtype, min, max, corr_plus, corr_minus
      "browse" ( vagy "rows"):  sorok listázása   (cols: oszlopszűrés,  query: sorszűrés)
      "plot":             összes dtype="num" és distinct_values>1 oszlop nyomtatása egy közös diagramba (mindegyik normalizálva, "scatter gauss")
                            - kihagyja a számított oszlopokat (corr=1 vagy -1 egy előtte lévővel)
                            - oszlopindex-lista állhat mögötte (oszlop-sorszámok, lásd "info")
                            - groupby megadása esetén a distinct értékekre külön vonaldiagrammok (mezőnként és groupby értékekenként)
      "2":                második oszlop értékeloszlása  (series:  value, count,  leggyakoribb legelöl)
      "2 plot":           második oszlop nyomtatása (scatter gauss)     
      "2 corr":           második oszlop korrelációi a többi oszloppal  (plot)  Az oszlopsorszám a tbl összes oszlopára vonatkozik (lásd még "corr 2")
      "corr":             korrelációs táblázat megjelenítése (szöveges)
      "corr 2":           a korrelációs táblázat második oszlopának korrelációi a többi oszloppal (lásd még: "2 corr")
      "corrplot":         az összes korreláció megjelenítése diagrammon
      "corrmax":          legnagyobb pozitív korrelációk megjelenítése (szöveges)     "corrmax 2" - második oldal ...
      "corrmin":          legnagyobb negatív korrelációk megjelenítése (szöveges).    "corrmin 2" - második oldal ...

      "drop 1,2,5":       oszlopok törlése (utána az "info" ismételt megjelenítése)
      "rename 3 code2":   oszlop átnevezése (egy-szavas oszlopnév adható meg, ékezetes betűk megengedettek; utána újra "info)
 
    cols: oszlopnevek rövidített felsorolása. Ha üres, akkor az összes számoszlop (számított másodlagos oszlopok nélkül) 
            példa:  "total case,cumulative":  van benne "total" és "case", vagy van benne "cumulative"  (több oszlop lehet az eredmény)
            Oszloponként külön lineplot, alapesetben ugyanabban a diagramban (ha groupby is többes, akkor direktszorzat)
            Subplot határoló: //   Ha a groupby-nál is meg van adva, akkor a groupby-nak van prioritása (összehangolandó a kettő)
            Hatással van:  plot, rows, corr, corrplot, corrmax, corrmin 
    query:  alapszűrő (nem kötelező)  pl. 'Dátum>="2021.01.01"'    'Continent=="Europe"'    'Year>1900 and Year<=2020'   'Continent.str.startswith("Eu",case=False)'
    groupby:  külön browse vagy plot a groupby oszlop distinct értékeire (csoportosító szűrésről van szó, nem aggregálásról) 
            'Country'     külön leválogatás / diagram-vonal a 'Country' oszlop összes distinct értékére         
            'Country:hun,cze,united king,slo,rom'     a 'Country' oszlop megadott szűrőknek megfelelő értékeire (lásd: filternames())
            'Country:hun,cze,pol,rom//germ,austria,france'     két subplot (nrows,ncols paraméterekkel állítható be a sublopotok pozíciója)
    orderby:  oszlopnév vagy oszlopnevek felsorolása (lehet gyorsírásos is, lásd cols, NOT nélkül)). 
            A legvégére "desc" írható (sajnos több oszlopos esetén is csak globálisan adható meg)

    toexcel:  írja ki excel fájlba   ("info", "browse", "5", "corr", "corrmin", "corrmax")
    plotparams:  FvPlot, plotinit és plotshow összes lehetséges argumentuma
            Legyakoribb:  suptitle,title,ylabel,plttype,label,annotate,normalize,color
            Ha a groupby-ban subplot-ok vannak megadva, akkor // határolással subplot-szintű paraméterek adhatók meg 
                (ha nincs elég, akkor az utolsó érvényes a többi subplot-ra) 

    return: tbl (colname,dtype,not_null,unique,...)     Rendezés: filled (desc),  repeat (asc)
    """

    tblOut=None

    if isint(todo): todo=str(todo)

    words=todo.split()
    if len(words)==0: return
    firstword=words[0]

    if query:
        tbl=tbl.query(query)


    groupcol=''
    groups=[]
    subgroups=[['']]
    if groupby:
        subgroups=[]
        groupcol,groupfilter = splitfirst(groupby,':')
        groupcol=groupcol.strip()
        if groupfilter: groupfilter=groupfilter.strip()

        # A táblában előforduló összes lehetséges csoportosító érték bekérése
        groups=tbl[groupcol].unique()       # distinct
        if groupfilter:
            subgroups=groupfilter.split('//')             # Country:hun,pol//german,france       //-határolással külön subplot-ok
            for i in range(len(subgroups)):
                subgroups[i]=filternames(groups,subgroups[i])      # helper függvény
                # minden subgroup egy lista a tbl-ben lévő oszlopnevekkel  (az oszlopnevek elején '^' állhat, ami kiemelést jelez)
        else:
            subgroups.append(groups)

    if cols: 
        # groupcol felvétele az oszloplistába (a szűréshez szükség lesz rá)
        if groupcol:
            partsL=cols.split(' NOT ')
            if not groupcol in partsL[0]: 
                cols=partsL[0].strip() + ',' + groupcol
                if len(partsL)>1: cols=cols + ' NOT ' + partsL[1]
        tbl=tblfiltercols(tbl,cols)

    if orderby:
        ascending=True
        if endwith(orderby,' desc'):
            orderby=cutright(orderby,' desc')
            ascending=False
        orderbycols=filternames(list(tbl),orderby)      # helper függvény
        if len(orderbycols)>0:
            tbl=tbl.sort_values(orderbycols,ascending=ascending)

        # # Oszloponként külön-külön rendezem  (van többoszlopos arg-változat is, de a desc csak globálisan adható meg)
        # NEM JÓ.  Ez egymást követő rendezések nem veszik figyelembe a már beállított sorrendet (talán copy kellene, de az időigényes lehet)
        # for orderbycol in reversed(orderbycols):
        #     if endwith(orderbycol,' desc'):
        #         orderbycol=cutright(orderbycol,' desc')
        #         tbl=tbl.sort_values(orderbycol,ascending=False)
        #     else:
        #         tbl=tbl.sort_values(orderbycol)


    rowcount=len(tbl)
    colnames=list(tbl)
    colcount=len(colnames)
    dtypes=tbl.dtypes

    # Oszlopok kivonatos adatai
    if todo in ['info','minmax']:
        tblcorr=tbl.corr()
       
        out=('Rekordok száma: ' + str(rowcount) + '\n' +
            'Oszlopok száma: ' + str(colcount))
    
        colnames_all=['index'] + colnames
        
        aOut=[]
        nColindex=0
        for colname in colnames_all:
            #print(colname)
            if colname=='index': 
                sorszam='index'
                col=tbl.index
                dtype=str(tbl.index.dtype)
                colnameout=tbl.index.name       # nem mindig van neve
                if colnameout==None: colnameout=''
            else: 
                sorszam=str(nColindex)
                nColindex+=1
                col=tbl[colname]
                dtype=dtypes[colname].name
                colnameout=colname
            
            if dtype=='object': dtype='string'          # vegyes is lehet (ha egyetlen None előfordul, akkor is ide kerül)
            elif beginwith(dtype,'float'): dtype='float'
            elif beginwith(dtype,'int'): dtype='int'
            elif beginwith(dtype,'decimal'): dtype='dec'
            elif beginwith(dtype,'datetime'): dtype='datetime'      # nincs automatikus parse
        
            if rowcount==0: 
                if todo=='info': aOut.append((sorszam,colnameout,dtype,'','','','',''))
                elif todo=='minmax': aOut.append((sorszam,colnameout,dtype,'','','',''))
                continue

            # üres értékek, feltöltöttség
            if todo=='info':
                if colname=='index': nulldb=0
                else: nulldb=col.isnull().sum()       # None, np.nan 
                nulladb=nulldb
                if dtype=='string': nulladb += (col=='').sum() + (col=='0').sum()
                elif dtype in ['float','int','dec']: nulladb += (col==0).sum()
                elif dtype=='bool': nulladb += (col==False).sum()

                if nulldb==0: notnull='100% (all)'
                else: notnull = '{:.2%}'.format((rowcount-nulldb) / rowcount)

                if nulladb==0: notnulla='100% (all)'
                else: notnulla = '{:.2%}'.format((rowcount-nulladb) / rowcount)


                # ismétlődő értékek száma
                repeat = len(tblduprows(tbl,colname,False))         # colname=="index" esetén is működik
                if repeat==0: unique='100% (unique)'
                else: unique='{:.2%}'.format((rowcount-repeat)/rowcount)
            
                # leggyakoribb érték
                distinct=0
                sample=''
                ser_valuecounts=col.value_counts()      # series     
                if len(ser_valuecounts)>0:
                    sample=ser_valuecounts.index[0]
                    samplecount=ser_valuecounts.iloc[0]
                    distinct=len(ser_valuecounts)
                    if distinct==rowcount: distinct=str(distinct) + ' (unique)'
                    if dtype in ['float','int','dec']: sample='{:,.6g}'.format(sample)
                    else: sample=sample=txtshorten(str(sample),30)
                    sample='"' + sample + '" (' + str(samplecount) + ')' 

                aOut.append((sorszam,colnameout,dtype,notnull,notnulla,unique,distinct,sample))
            
            elif todo=='minmax':
                ser=col.dropna()               # None értékek kihagyása (enélkül hibaüzenet jelenhet meg)
                min=ser.min()                           # számok esetén az nan kimarad, de a 0 benne van
                if dtype not in ['float','int','dec']: min='"' + txtshorten(str(min),30) + '"'
                max=ser.max()
                if dtype not in ['float','int','dec']: max='"' + txtshorten(str(max),30) + '"'
                avg=''
                if dtype in ['float','int','dec','bool']: avg=ser.mean()         # bool esetén is értelmezhető
        
                corr_plus=''
                corr_minus=''
                if dtype in ['float','int','dec','bool']:
                    ser=tblcorr[colname]             # azok a korrelációk, amelyekben az aktuális oszlop szerepel
                    ser=ser[ser.index!=colname]         # saját magával nem kell korreláció
                    ser_plus=ser[ser>0.1]
                    if ser_plus.count()>0: 
                        corr_plus=ser.idxmax() + ' (' + '{:.3g}'.format(ser.max()) + ')'
                    ser_minus=ser[ser<-0.1]
                    if ser_minus.count()>0: 
                        corr_minus=ser.idxmin() + ' (' + '{:.3g}'.format(ser.min()) + ')'

                aOut.append((sorszam,colnameout,dtype,min,max,corr_plus,corr_minus))
    
        if todo=='info':
            tblOut=TblFromRecords(aOut,'x,colname,dtype,not_null,not_null_0,uniqueness,distinct_values,most_frequent_sample','x')
        elif todo=='minmax':
            tblOut=TblFromRecords(aOut,'x,colname,dtype,min,max,corr_plus,corr_minus','x')
    
        print(out + '\n\n' + str(tblOut) + '\n')
    
    # Browse (listázás)
    elif firstword in ['browse','rows']:
        if toexcel: tblOut=tbl      # todo:  külön táblázatokba kellene beírni (azonos munkalapon vagy külön munkalapokon)
        else: 
            if len(subgroups)==0:    # ha nincs megadva group
                print(tbl)
            else:
                for subgroup in subgroups:          # minden subgroup egy lista (egyetlen '' is lehet benne, ha nincs groupby)
                    for groupvalue in subgroup:         # vonalak csoportosító értékenként és oszloponként 
                        if groupvalue!='': tblL=tbl.loc[tbl[groupcol]==groupvalue]
                        else: tblL=tbl
                        print('\n\n' + groupvalue.upper())
                        print(tblL)


    # Plot
    elif firstword in ['plot','interpolate']:
       
        # megadható egy oszlopssorszám felsorolás   (pl.  "plot 1,3,5";  a cols argumentum-ban is megadhatóak az oszlopnevek)
        colindexes=None
        if len(words)>1: colindexes=words[1].split(',')

        # dtype="num" és distinct_values>1 oszlopok begyűjtése (a számított oszlopok nélkül)
        tblcorr=tbl.corr()
        colnamesL=[]
        colindexL=[]
        for i in range(len(colnames)):
            if colindexes and not (str(i) in colindexes): continue      # ha van explicit colindex felsorolás, akkor csak azon belül
            colname=colnames[i]
            dtype=dtypes[colname].name
            if not (dtype in ['float64','int64']): continue
            distinct=len(tbl[colname].value_counts())
            if distinct<2: continue
            bCalc=False
            strL=''
            corrindex=tblcorr.index.get_loc(colname)
            for j in range(corrindex):          # előtte lévők
                c=tblcorr.iloc[j,corrindex]
                strL=strL + str(c) + ','
                if np.isclose(c,1): bCalc=True
            #print('corrindex:' + str(corrindex) + '  ' + strL)
            if bCalc: continue
            colnamesL.append(colname)
            colindexL.append(i)
        
        # FvPlot paraméterek    
        normalize_default=1
        if len(colnamesL)==1: normalize_default=None
        dsetsoft(plotparams,normalize=normalize_default)
 
        dsetsoft(plotparams,annotate='localmax2 last',area=True)

        plttype_default='gauss'
        nLines=len(colnamesL)*len(subgroups)
        if nLines==1: plttype_default='scatter gauss'
        dsetsoft(plotparams,plttype=plttype_default)

        # pltinit paraméterek
        dsetsoft(plotparams,suptitle='Diagramok (tblinfo)',height=0.8,left=0.07,right=0.84,bottom=0.09)
        
        title_in=plotparams.get('title')     # lista is lehet  (több subplot esetén egyedi feliratok)
        if type(title_in)==str: title_in=[title_in]     # egységesen lista

        # title=''
        # if len(groups)==1: title=groups[0]
        # if len(groups)>0:
        #     if len(groups)<5: title=txtshorten(str(groups),80) 
        #     else: title=str(len(groups)) + ' adatkategória'
        # if len(colnamesL)<3: title=title + '  ' + txtshorten(str(colnamesL),50) 
        # else: title=title + '  ' + str(len(colnamesL)) + ' adatfajta'
        # title=title.lstrip()
        # dsetsoft(plotparams,title=title)

        # pltshow paraméterek
        dsetsoft(plotparams,annot_fontsize=8)

        # Ha subplotok vannak, akkor mindenképpen kell nrows és ncols
        nSubDb=len(subgroups)
        nrows=plotparams.get('nrows')
        ncols=plotparams.get('ncols')
        if nSubDb>1 and (not nrows or not ncols):        
            if not nrows: 
                if ncols and ncols>0: nrows=int(math.ceil(nSubDb/ncols))
                else: nrows=int(math.floor(math.sqrt(nSubDb)))
            if not ncols: ncols=int(math.ceil(nSubDb/nrows))
            dsetsoft(plotparams,nrows=nrows,ncols=ncols)

        colors_in=plotparams.get('colors')
        annotcolor_in=plotparams.get('annotcolor')

        if firstword=='plot': pltinit(**plotparams)
        
        # Vonaldiagramok rajzolása
        labelToSeries={}
        aGausswidth=[]              # tájékoztató felirat lesz a jobb felső sarokban (plotonként eltérő lehet, ezért számítani kell)
        for nSub in range(len(subgroups)):          # minden subroup egy lista (egyetlen '' is lehet benne, ha nincs groupby)
            subgroup=subgroups[nSub]
            
            # Title beállítása
            title=''
            if title_in:
                if nSub<len(title_in): title=title_in[nSub]
                else: title=title_in[-1]                        # ha nincs ilyen indexű title_in, akkor az utolsó 
            # default beállítás
            if title=='':
                if len(subgroup)==1: title=subgroup[0]      # ha egyetlen csortosító érték van a subgroup-ban
            if title: plotparams['title']=title

            if len(subgroups)>1 and firstword=='plot': 
                pltinitsub(nSub,**plotparams)

            for groupvalue in subgroup:         # vonalak csoportosító értékenként és oszloponként 
                if len(groupvalue)>0 and groupvalue[0]=='^':
                    groupvalue=cutleft(groupvalue,'^')
                    plotparams['colors']={'color':'orange', 'linewidth':1.5, 'alpha':1}
                    plotparams['annotcolor']='kiemelt'
                else: 
                    plotparams['colors']=colors_in
                    plotparams['annotcolor']=annotcolor_in
                
                if groupvalue!='': tblL=tbl.loc[tbl[groupcol]==groupvalue]
                else: tblL=tbl
                if plotparams.get('plttype') and ('gauss' in plotparams.get('plttype')): 
                    aGausswidth.append(plotparams.get('gausswidth',len(tblL)/10))   # nem garantált, hogy minden vonalra azonos (az átlag fog megjelenni kivonatos adatként)

                nColnames=len(colnamesL)
                for i in range(nColnames):
                    colname=colnamesL[i]
                    # label kitalálása
                    if (groupvalue=='') or len(subgroup)==1: caption=colname
                    else:
                        if nColnames==1: caption=groupvalue
                        else: caption=str(colindexL[i]) + ' ' + groupvalue + '_' + colname
                    if firstword=='plot':
                        FvPlot(tblL[colname],label=txtshorten(caption,12),**plotparams)
                    elif firstword=='interpolate':
                        labelToSeries[caption]=FvPlot(tblL[colname],seronly=True,**plotparams)

            
            if len(subgroups)>1 and firstword=='plot': 
                pltshowsub(**plotparams)


        # aGausswidth=[]
        # if len(groups)>0:
        #     aAnnotL=[]
        #     for group in groups:
        #         tblL=tbl.loc[tbl[groupcol]==group]
        #         print('group:' + str(group) + '  len(tblL):' + str(len(tblL)))
        #         if plotparams.get('plttype') and ('gauss' in plotparams.get('plttype')): 
        #             aGausswidth.append(plotparams.get('gausswidth',len(tblL)/10))   # nem garantált, hogy minden vonalra azonos (az átlag fog megjelenni kivonatos adatként)
        #         nColnames=len(colnamesL)
        #         for i in range(nColnames):
        #             colname=colnamesL[i]
        #             if nColnames==1: caption=group
        #             else: caption=str(colindexL[i]) + ' ' + group + '_' + colname
        #             FvPlot(tblL[colname],label=caption,**plotparams)
        # else:
        #     if plotparams.get('plttype') and ('gauss' in plotparams.get('plttype')): 
        #         aGausswidth.append(plotparams.get('gausswidth',len(tbl)/10))   # kivonatos adatként fog megjelenni
        #     for i in range(len(colnamesL)):
        #         colname=colnamesL[i]
        #         FvPlot(tbl[colname],label=str(colindexL[i]) + ' ' + colname,**plotparams)
        
        # információk a jobb felső sarokban
        lines=[]
        # Ha a tbl-nek van source adata, akkor kiírja  (egyedi módon állítható be a beolvasáskor)
        try:
            if tbl.attrs['source']: lines.append('Forrás: ' + tbl.attrs['source'])
        except: pass
        if len(aGausswidth)>0: lines.append('Gauss mozgóátlag effektív szélessége: ' + '{:.3g}'.format(np.average(aGausswidth)/2))  # effektív szélesség
        annottexts=None
        if len(lines)>0: 
            annottext={'x':0.95,'y':0.97,'caption':joinlines(lines),'ha':'right','va':'top','fontsize':7,'alpha':0.5}
            #plt.text(1,1.12,joinlines(lines),ha='right',va='top',fontsize=7,alpha=0.5,transform = plt.gca().transAxes)
            if plotparams.get('annottexts'): plotparams['annottexts'].append(annottext)
            else: plotparams['annottexts']=[annottext]
                             
        #print('pltshow')
        if firstword=='plot': pltshow(**plotparams)


    # info=5   info='5'
    elif isint(todo):
        ncol=int(todo)
        if ncol>=rowcount:
            print('Hiba: tblinfo, nincs ilyen indexű oszlop (' + str(ncol) + '), ezért nem jelenthető meg értékeloszlás\n' +
                  'Az elérhető oszlopindexeket a "tblinfo(tbl)" utasítással lehet lekérdezni.')
            return
        colname=colnames[ncol]
        firstword=colname    # excel-be íráskor lehet szükséges
        tblOut=tbl[colname].value_counts().reset_index(drop=False).sort_values([colname,'index'],ascending=[False,True]).set_index('index')
        print(tblOut)
        #print('"' + tbl.columns[ncol] + '" oszlop értékeloszlása:\n')
        #print(tbl[colnames[ncol]].value_counts())
    
    # info='corr'
    elif len(words)==1 and words[0]=='corr':
        tblOut=tbl.corr()
        print(tblOut)

    # info='corrplot'
    elif todo=='corrplot':
        tblcorr=tbl.corr()
        numcols=len(list(tblcorr))
        if numcols<=5: format='simple'
        else: format='normal'
        FvCorrPlot(tbl,format)
        return
            
    # info='5 corr'
    elif len(words)==2 and isint(words[0]) and words[1]=='corr' and int(words[0])<colcount:
        colname=colnames[int(words[0])]
        pltinit(suptitle='Korreláció',title=colname,left=0.4,right=0.9,width=0.5,height=0.8)
        tblcorr=tbl.corr()
        sb.heatmap(tblcorr[[colname]].sort_values(by=colname, ascending=False)[1:], vmin=-1, vmax=1, annot=True, cmap='BrBG')
        pltshow()

    # info='corrmax',  info='corrmin'
    elif firstword in ['corrmax','corrmin']:
        tblcorr=tbl.corr()
        ascending = (todo=='corrmax')
        nlen=len(tblcorr)
        aRec=[]
        for i in range(nlen):
            for j in range(i+1,nlen):
                aRec.append((tblcorr.index[i],tblcorr.columns[j], tblcorr.iloc[i,j]))
        tblOut=TblFromRecords(aRec,'adat1,adat2,corr')
        tblOut=tblOut.sort_values(by='corr',ascending=ascending)
        if len(words)==2 and isint(words[1]):
            nPage=int(words[1])
            if nPage<0:
                print(str(tblOut.tail(30)) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' 2")')        
            else:
                i=(nPage-1)*30
                print(str(tblOut.iloc[i:(i+30)]) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' ' + str(nPage+1) + '")')
        else:  
            print(str(tblOut.head(30)) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' 2")')        

           
    # info='drop 4'
    elif len(words)>=2 and words[0]=='drop':
        dropped=0
        for i in range(1,len(words)):
            if isint(words[i]) and int(words[i])<colcount: 
                tbl.drop(tbl.columns[int(words[i])], axis=1,inplace=True)
                dropped+=1
            else: 
                print('Hiba: tblinfo, "drop" után oszlopsorszámo(ka)t kell megadni')
        if dropped>0:
            print('tblinfo, ' + str(dropped) + ' oszlop lett törölve a táblázatból')
            tblinfo(tbl)

    # info='rename 6'
    elif len(words)==3 and words[0]=='rename' and isint(words[1]) and int(words[1])<colcount:
        tbl.rename(columns={tbl.columns[int(words[1])]: words[2]},inplace=True)
        print('Oszlop átnevezve\n')
        tblinfo(tbl)

    else:
        print('Hiba:  tblinfo,  oszlopsorszám, "drop" "rename" "corr" adható meg első szóként')
    

    if toexcel and len(tblOut)>0:
        path=nextpath('tblinfo_' + firstword + ' (%s).xlsx')
        try: 
            tblOut.to_excel(path)
            print('Excel fájlba írás: ' + path)
        except:
            print('A fájlba írás nem sikerült')

    if firstword=='interpolate':
        return labelToSeries    # A kirajzolt vonalakhoz tartozó series objektumok (felhasználható pl. interpolációra (servalueA))
    elif firstword=='plot':
        config.tblinfoplotsG = labelToSeries     


def tbltoexcel(tbl):
    # Excel fájlba írás, a Letöltések mappába
    path=nextpath('tbl browse (%s).xlsx')
    try: 
        tbl.to_excel(path)
        print('Excel fájlba írás: ' + path)
    except:
        print('A fájlba írás nem sikerült')



def tblfiltercols(tbl,colnamefilter):
    '''
    Oszlopszűrő az oszlopnevekben előforduló szövegrészekre. 
    - keresés szavanként (ÉS kapcsolat, a szavak sorrendje érdektelen)
    - vesszővel elválasztva OR feltételek
    - NOT után kizáró feltétel
      példa:  "total case,cumulative NOT smoothed":  van benne "total" és "case", vagy van benne "cumulative" és nincs benne "smoothed"
    Return:  tbl_filtered   (nem inplace)
    '''
    colnamesout=filternames(list(tbl),colnamefilter)
    if colnamesout:
        return tbl[colnamesout]
    

def filternames(names,filter):
    '''
    Szűrés a megnevezésekben előforduló szövegrészekre. 
    names:  választható nevek  (a kimenetbe ezek a nevek kerülhetnek)  
       list    vagy str (vesszős felsorolás)    vagy np.ndarray (np list)
       példa: ["cases", "total_cases", "cases_smoothed", "total_cases_smoothed"]
              "cases,total_cases,cases_smoothed"
       - a return mindenképpen list
    filter:    ha üres, akkor return names (nincs szűrés)
    - egy egyszerű vesszős névfelsorolás is megadható (case-insensitive, bárhol a szövegben)
    - általános esetben:
        - vesszővel elválasztva OR feltételek
        - OR feltételeken belül:  keresés szavanként (ÉS kapcsolat, a szavak sorrendje érdektelen)
        - NOT után kizáró feltétel
      példa:  "total case,cumulative NOT smoothed":  van benne "total" és "case", vagy van benne "cumulative" és nincs benne "smoothed"
              "NOT smoothed":    az összes, kivéve a "smoothed" szövegrészt tartalmazók
    - a kimeneti nevek sorrendje a filter-hez igazodik
    - a kiemelés jelzések ("^") a kereséskor levágva, majd a kimeneti listába is bekerülnek az oszlopnevek elé  (pl. "^hun,ger,^slovakia,austria")
        Az "asc" és a "desc" zárószavakat is levágja, és beírja a kimeneti lista oszlopnevei után 
    Return:  List, a filternek megfelelő nevek listája
    '''
    if type(names)==np.ndarray: names=list(names)
    elif type(names)==str: names=names.split(',')
    
    filter=filter.strip()
    if filter=='':
        return names

    if filter.startswith('NOT '):
        parts=filter.split('NOT ')
    else:
        parts=filter.split(' NOT ')
    if len(parts)>2: parts=parts[:2]        # part[0]: pozitív feltétel,  part[1]: negatív feltétel
    
    aNamesout=[]
    bNot=False
    for nPart in range(len(parts)):     # part[0]: pozitív feltétel,  part[1]: negatív feltétel
        part=parts[nPart]
        if part=='': 
            if nPart==0: aNamesout=names.copy()
            continue
        
        bKiemelt=False
        
        # vesszővel elválasztva VAGY feltételrészek
        orfilters=part.lower().split(',')
        orfilters_kiemelt=[False]*len(orfilters)

        # szóközzel elválasztva ÉS feltételrészek
        # for i in range(len(orfilters)): 
        #     # az orfilter elején lehet '^' karakter, ami kiemelést jelez
        #     if orfilters[i][0]=='^':                
        #         orfilters[i]=cutleft(orfilters[i],'^')
        #         orfilters_kiemelt[i]=True
        #     # ÉS minták, listává alakítás (egyelemű is lehet)
        #     orfilters[i]=orfilters[i].split()   # orfilters[i]: ["total","cases"]


        # ciklus a filterben szereplő mintákra  (a kimeneti lista sorrendje a filter-hez igazodik)
        for i,orfilter in enumerate(orfilters):    # orfilter:  ["hun"]   ["total","cases"]
            if len(orfilter)==0: continue
            bKiemelt=False
            if orfilter[0]=='^':                
                orfilter=cutleft(orfilter,'^')
                bKiemelt=True
            bDesc=False
            if endwith(orfilter,' asc'): orfilter=cutright(orfilter,' asc')
            elif endwith(orfilter,' desc'): 
                orfilter=cutright(orfilter,' desc')
                bDesc=True
            
            # szóközzel elválasztott keresőszavak, listává alakítás ("ÉS" kapcsolat, egyelemű is lehet)
            orfilter=orfilter.split()   # orfilter: ["total","cases"]
            
            
            # ciklus a választható nevekre
            for name in names:
                nameL=name.lower()
                bOkL=True
                for sample in orfilter:         # ÉS minták    
                    if not (sample in nameL):   # bárhol előfordulhat a névben
                        bOkL=False              # ha van olyan ÉS minta, ami nem fordul elő, akkor nincs találat
                        break
                # Ha a név megfelel az orfilter-nek
                if bOkL:
                    if bKiemelt: name='^' + name        # a kimeneti listába is kerüljön be a '^' jel
                    if bDesc: name = name + ' desc'
                    if nPart==0: 
                        if not (name in aNamesout): aNamesout.append(name)
                    elif nPart==1: 
                        if name in aNamesout: aNamesout.remove(name)
            

        # ELAVULT   Ez a változat a names-hez igazította a kimeneti lista sorrendjét
        # # ciklus a választható nevekre (ebben kell keresni)
        # for name in names:      
        #     nameL=name.lower()
        #     bOk=False
        #     bKiemelt=False
        #     for i in range(len(orfilters)):  # orfilter: ["total","cases"]
        #         orfilter=orfilters[i]
        #         if len(orfilter)==0: continue

        #         bOkL=True
        #         for sample in orfilter:         # ÉS minták    
        #             if not (sample in nameL):   # bárhol előfordulhat a névben
        #                 bOkL=False              # ha van olyan ÉS minta, ami nem fordul elő, akkor nincs találat
        #                 break
        #         if bOkL:
        #             bOk=True            # ha legalább egy OR-minta rendben van, akkor találat
        #             bKiemelt=orfilters_kiemelt[i]
        #             break
        #     if bOk: 
        #         if nPart==0: 
        #             if bKiemelt: aNamesout.append('^' + name)
        #             else: aNamesout.append(name)
                    
        #         elif nPart==1: 
        #             if name in aNamesout: aNamesout.remove(name)

    return aNamesout

def tblgroupbyvalues(tbl,groupcol,groupfilters=None):
    ''' GroupBy értékek listáját adja vissza. A lista ismeretében egy ciklus és a tábla szűrése a groupby értékekre
            for group in groups:
                tblL=tbl.loc[tbl[groupcol]==group]
    tbl:
    groupcol: colname a tbl-ben  (általában véges sok kategorizáló értékkel)
    groupfilters:  lásd filternames()    Ha üres, akkor a groupcol-ban előforduló összes érték. 
        keresőszavak felsorolása (köztük szóköz),  vesszővel elválasztva VAGY feltételek, 
           a végén NOT után olyan szórészletek, amelyek nem fordulhatnak elő a csoportosító értékben
    '''
    values=tbl[groupcol].unique()      # distinct

    # nan, None, '' ne kerüljön bele
    groupvalues=[]
    for value in values:
        if pd.isna(value) or value==None or value=='': continue
        groupvalues.append(value)

    if groupfilters:
        groupvalues=filternames(groupvalues,groupfilters)


    return groupvalues



def serdtype(tblorser,colname='values'):
    '''
    colname:  ser esetén "index" vagy "values" lehet;   tbl esetén "index" vagy oszlopnév lehet
    return:  "string"  "num"   "datetime"
    '''
    dtype=''
    if colname=='index':
        dtype=tblorser.index.dtype
    elif colname=='values':
        dtype=tblorser.dtype
    else:
        dtype=tblorser[colname].dtype

    if dtype=='object': dtype='string'          # vegyes is lehet (ha egyetlen None előfordul, akkor is ide kerül)
    elif dtype in ['float64','int64']: dtype='num'      # nan lehet benne
    elif dtype=='datetime64[ns]': dtype='datetime'      # nincs automatikus parse

    return dtype



def tblduprows(tbl,checkcol,keepfirst=False):
    # Egy táblázatot ad vissza:  minden olyan rekord, amelyre az adott mező értéke ismétlődő
    # A duplikált sorok kidobása:  
    # checkcol: 'index'  /   oszlopnév     /    több oszlopnév listaként
    #     'index':  a kulcsértékek duplikációi   (a pandas-ban nem garantált a kulcsértékek unicitása) 
    if keepfirst: keep='first'
    else: keep=False
    
    if checkcol=='index': return tbl[tbl.index.duplicated(keep=keep)]
    else: return tbl[tbl[checkcol].duplicated(keep=keep)]                # .sort_values(by=colname)

def tbldropduprows(tbl,checkcol):
    # Kidobja a másodlagos ismétlődő sorokat (minden csoportból az elsőt tartja meg)
    #   Nincs inplace változat
    # colname: mire ellenőrizze az unicitást     "all"   /   "index"    /    oszlopnév     /      oszlopnevek listája
    if checkcol=='index':
        return tbl[~tbl.index.duplicated()]
    elif checkcol=='all':
        return tbl[~tbl.duplicated()]
    else:
        return tbl[~tbl[checkcol].duplicated()]

def tblsetindex(tbl,column,inplace=True,drop=False,checkunique=True):
    dtype=serdtype(tbl,column)
    tbl.set_index(column,verify_integrity=checkunique,drop=drop,inplace=inplace)
    if not drop: tbl.index.name='index'          # fontos, mert egyébként névütközés lenne a megmaradó oszloppal
    if dtype=='datetime': tbl.index=pd.to_datetime(tbl.index)     # elvileg automatikus is lehetne


def serduprows(ser,bIndex=True):
    # egy ser-t ad vissza:    minden olyan xy, amire az y ismétlődő
    # bIndex:  True esetén az index duplikációit listázza
    if bIndex: return ser[ser.index.duplicated(keep=False)]
    else: return ser[ser.duplicated(keep=False)]                      # .sort_values()  

def serdropduprows(ser,bIndex=True):
    # Kidobja a másodlagos ismétlődő sorokat (minden csoportból az elsőt tartja meg)
    # colname: mire ellenőrizze az unicitást     "all"   /   "index"    /    oszlopnév     /      oszlopnevek listája
    if bIndex:
        return ser[~ser.index.duplicated()]
    else:
        return ser[~ser.duplicated()]



def FvWeightedAvg(ser,serWeights):
    # elvárás: a két sorozat x értékei össze vannak hangolva  (pl. ugyanahhoz a dataframe-hez taroznak)
    # az nan értékeket nem veszi figyelembe (kihagyás, ha akár az érték, akár a súly nan)
    # Két tömb esetén:  np.average(aIn,aWeights)
    aL=ser.values
    aW=serWeights.values
    sum=0
    sumW=0
    for i in range(len(aL)):
        if np.isnan(aL[i]) or np.isnan(aW[i]): continue
        sum+=aL[i]*aW[i]
        sumW+=aW[i]
    if sumW: return sum / sumW


def FvPlot(ser,plttype='scatter gauss',label='',seronly=False,annotate='localmax last',normalize=None,area=False,colors=None,axindex=None,**params):
    ''' Series nyomtatása pont-diagramként / interpolálva / mozgóátlaggal / spline illesztéssel  (több típus is kérhető)
     A vonal(ak) annotálása is kérhető (a felirat label lehet egyedi,  pozíció:  last, first, max, min, konkrét x-érték idézőjelben)
     return (config.serplotlastG):  a transzformált és kinyomtatott series
    
     plttype:    megjelenítési típusok felsorolása (szóköz határolás, nem kell vessző)
       original:   eredeti függvénygörbe  (NaN pontok megszakítják a görbét)
       scatter:    pontok megjelenítése
       interpolated:   nan értékek helyébe lineáris illesztés
       gauss:      gaussAvg megjelenítése  (előtte interpoláció az nan értékekre; ha meg van adva a resample, akkor előtte linear resample)
       gauss+spline:    egymás utáni alkalmazás
       gaussgrad:  derivált megjelenítése  (előtte és utána is gauss simítás)
       linear:     egyenes illesztés (legkisebb négyzetek módszere)
       resample:   linear resample  (az eredeti pontok összekötése egyenesekkel, majd a megadott frekvenciának megfelelő mintavételek)
       spline:     linear resample + spline 
     label:        legend-ben megjelenő felirat
     seronly:      ne rajzoljon, csak az interpolált kimeneti series-t adja vissza (több plttype esetén az utolsónak megfelelő series)
     annotate:     'last'  'first'  'max'  'min' 'localmax', 'localmin', konkrét x-érték (pl.'20210101', '12';  az x értéknek nem kell szerepelnie a ser x-tömbjében (interpoláció))    
                   - szóközzel elválasztva több is megadható, pl. "last max'    (lásd még: annotplus - pontfeliratozások)
                   - localmax:  legalább 5 pont szélességű lokális maximum, ha több van, akkor a legnagyobb y értékű
                   - localmax2:  max 2 localmax
                   - ha több vonalat is rajzol a függvény (pl. 'spline guass'), akkor az utolsó görbére érvényesül
     normalize:    maxOut adható meg (általában 1; az abszolút értékre vonatkozik;  0 vagy None esetén nincs normalizálás)
     area:         True esetén szürke kitöltés az x-tengelytől (vonal rajzolás nélkül; scatter-nél érdektelen)
     axindex:      csak subplotok esetén szükséges. Subplot esetén a rajzterület indexe
     colors:       {color,linewidth,alpha}   példa: {'color':'orange', 'linewidth':2, 'alpha':0.6}
                   - ha nincs megadva, akkor default (figyelembe véve a speccolors argumentumot is, amiben a label-hez tartozó speciális színek lehetnek)
     params:       
       gausswidth      gauss    hány adatra terjedjen ki az átlagolás. Az effektív gauss szélesség ennek a fele. Default: a teljes x-szélesség tizede.
       trendpoint      gauss, spline:   (x,y) trendpont a jobb szélen (x legyen nagyobb minden meglévő x értéknél)
       resample        spline,  pl '100,500'   az első szám az előzetes lineáris resample sűrűsége, a második a kimeneti görbéé
                       linear   pl '200'    (nem kötelező; két szám esetén csak az első szám érdekes). Ne az eredeti x értékeke kerüljenek be a kimeneti ser-be, hanem egy egyenletes mintavétel
       extend          spline, linear  (csak resample esetén érvényesül;  hány ponttal menjen túl a bal illetve a jobb szélen. Példa: '0,30'
       extend_to       kitejesztés jobbra a megadott értékekig (ha nem terjed odáig). Az utolsó ismert adatot ismétli meg (konzervatív karakterű)
       extend_from     kiterjesztés balra.  Ha az x-tengelyen dátum van, akkor string-fromátumban is megadható  
       splinediff      spline   másként: accuracy

       annotatecaption fix string vagy str.format sablon.   Példa:  "{label} {plttype} ({y:.2%})" 
                       - ha nincs megadva, akkor label or plttype
                       - használható változónevek:  label, plttype (utolsó szó), x, y
       annotplus       egyedi pont-feliratozások, dictionary-ben megadva:  {x1:felirat1,x2:felirat2,...}
                       - az x értékek lehetnek dátumok is (stringként megadva). Ha nincs ilyen dátum a ser-ben, akkor interpoláció.
                       - a feliratok fix szövegek (nem sablon)
       annotcolor      az annotáció betűszíne  (pl. 'red')

       stdarea         True esetén szórás-sáv rajzolása (default False)
    '''

    if len(ser)==0: return

    ax=None
    if axindex: ax=plt.gcf().axes[axindex]

    annotcolor=None
    if not colors and config.linecolorsG:
        colors=config.linecolorsG.get(label) or {}
        if colors: colors=colors.copy()                   # ha globális változóban van tárolva, akkor a globális változó értéke ne változzon
    if not colors: colors={}
    # else: annotcolor=colors.pop('annotcolor',None)        # a colors-ból törli az annotcolor-t

    if normalize==0: normalize=None

    def sub_plotline(serplot,plttype):
        if serisempty(serplot):
            print('HIBA: FvPlot  Nincs nyomtatható adat   label=' + label + ' plttype:' + plttype)
        else:
            labelout=label or plttype
            if area:
                try:
                    plt.fill_between(serplot.index,serplot.values,0,alpha=0.08,color='0.3')
                except:
                    print('HIBA  FvPlot  Area plot  label='+ str(label) + '  plttype:' + str(plttype) )
                if colors=={}: colorsL={'color':'0.5', 'alpha':0}       # csak közvetlenül megadott szín vagy kiemelt label esetén nem üres
                else: colorsL=colors
                serplot.plot(label=labelout,ax=ax,**colorsL)    # enélkül nem jók az xtick feliratok és x irányban nem megy a széléig
            else: serplot.plot(label=labelout,ax=ax,**colors)

    extend_to=params.get('extend_to')
    if extend_to:
        ser=ser.copy()          # enélkül figyelmeztetések jelennek meg a futtatáskor
        if type(extend_to)==str: extend_to=pd.to_datetime(extend_to) 
        max_x=datefloat(max(ser.index))
        if max_x<datefloat(extend_to): ser[extend_to]=ser[max(ser.index)]
    extend_from=params.get('extend_from')
    if extend_from:
        if type(extend_from)==str: extend_from=pd.to_datetime(extend_from) 
        min_x=datefloat(min(ser.index))
        if min_x>datefloat(extend_from): ser[extend_from]=ser[min(ser.index)]

    serOut=None
    words=plttype.split()
    plttypelast=words[-1]
    if 'gauss' in words:
        resample=params.get('resample')
        if resample: serResample=FvLinearResample(ser,count=resample)
        else: serResample=ser
        gausswidth=params.get('gausswidth')
        trendpoint=params.get('trendpoint')
        serGauss=FvGaussAvg(serResample,windowwidth=gausswidth,trendpoint=trendpoint)
        if normalize: serGauss=normalizeSer(serGauss,normalize)
        if not seronly: sub_plotline(serGauss,'gauss')
        if plttypelast=='gauss': serOut=serGauss 
    if 'gauss+spline' in words:
        gausswidth=params.get('gausswidth')
        trendpoint=params.get('trendpoint')
        serGauss=FvGaussAvg(ser,windowwidth=gausswidth,trendpoint=trendpoint)
        resample=params.get('resample')
        extend=params.get('extend')
        splinediff=params.get('splinediff')
        serSpline=FvSmoothSpline(serGauss,resample=resample,extend=extend,diffByMaxmin=splinediff)    # csak a gauss-ra érvényesül a trendpoint
        if normalize: serSpline=normalizeSer(serSpline,normalize)
        if not seronly: sub_plotline(serSpline,'gauss+spline')
        if plttypelast=='gauss+spline': serOut=serSpline 
    if 'gaussgrad' in words:
        gausswidth=params.get('gausswidth') or int(len(ser)/10)
        serGauss=FvGaussAvg(ser,windowwidth=gausswidth)
        serGaussGrad=FvGradient(serGauss)
        serGaussGrad=FvGaussAvg(serGaussGrad,windowwidth=gausswidth*2)       
        if normalize: serGaussGrad=normalizeSer(serGaussGrad,normalize)
        if not seronly: sub_plotline(serGaussGrad,'gaussgrad')
        if plttypelast=='gaussgrad': serOut=serGaussGrad 
    if 'spline' in words:
        resample=params.get('resample')
        extend=params.get('extend')
        splinediff=params.get('splinediff')
        trendpoint=params.get('trendpoint')
        serSpline=FvSmoothSpline(ser,resample=resample,extend=extend,diffByMaxmin=splinediff,trendpoint=trendpoint)
        if normalize: serSpline=normalizeSer(serSpline,normalize)
        if not seronly: sub_plotline(serSpline,'spline')
        if plttypelast=='spline': serOut=serSpline 
    if 'linear' in words:
        resample=params.get('resample')
        extend=params.get('extend')
        serLinear=FvLinear(ser,resample=resample,extend=extend)
        if normalize: serLinear=normalizeSer(serLinear,normalize)
        if not seronly: sub_plotline(serLinear,'linear')
        if plttypelast=='linear': serOut=serLinear 
    if 'resample' in words:
        serResample=FvLinearResample(ser)
        if normalize: serResample=normalizeSer(serResample,normalize)
        if not seronly: sub_plotline(serResample,'resample')
        if plttypelast=='resample': serOut=serResample 
    if 'interpolated' in words:
        serInterpolated=(ser.interpolate(limit_area='inside'))
        if normalize: serInterpolated=normalizeSer(serInterpolated,normalize)
        if not seronly: sub_plotline(serInterpolated,'interpolated')
        if plttypelast=='interpolated': serOut=serInterpolated 
    if 'original' in words:
        if normalize: serOriginal=normalizeSer(ser,normalize)
        else: serOriginal=ser
        if not seronly: sub_plotline(serOriginal,'original')
        if plttypelast=='original': serOut=serOriginal
    if 'scatter' in words:
        if normalize: serScatter=normalizeSer(ser,normalize)
        else: serScatter=ser
        if not seronly: plt.scatter(serScatter.index,serScatter.values,label=(label or 'scatter'))
        if plttypelast=='scatter': serOut=serScatter

    if seronly:
        return serOut.copy()

    if not serisempty(serOut):
        #print('serOut:' + str(serOut))
        
        stdarea=params.get('stdarea')
        if stdarea:
            aYOut=servalueA(serOut,ser.index.array)
            if normalize: aY=normalizeSer(ser,normalize).array
            else: aY=ser.array

            sum=0
            for i in range(len(aY)):
                sum += (aYOut[i] - aY[i])**2
            std=math.sqrt(sum/len(aY))
            #print('std: ' + str(std))
            #print(list(zip(aY,aYOut)))

            plt.fill_between(serOut.index.array,(serOut+std).array,(serOut-std).array,color='silver',alpha=0.3)

        # Vonal-szintű annotációk
        if annotate:
            caption=label or plttypelast
            pattern=params.get('annotatecaption')           # {}-jeles string, lehetséges változónevek:  label, plttype, x, y
            words=annotate.split()

            annotcolor=params.get('annotcolor')        

            xfirst,yfirst=serfirstvalue(serOut)
            xlast,ylast=serlastvalue(serOut)

            # last, first, max, min, egyedi
            labels_x=[]         # 0-1 közötti label-pozíciók (ütközés-ellenőrzéshez kell a következő körben)
            for word in words:
                if word=='last': 
                    if xlast is not None:
                        if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=xlast,y=ylast)
                        FvAnnotateAdd(x=xlast,y=ylast,caption=caption,type='last',color=annotcolor)
                        labels_x.append(1)
                elif word=='first': 
                    if xfirst is not None:
                        if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=xfirst,y=yfirst)
                        FvAnnotateAdd(x=xfirst,y=yfirst,caption=caption,type='first',color=annotcolor)
                        labels_x.append(0)
                elif word=='max': 
                    x=serOut.idxmax()
                    y=serOut.loc[x]
                    if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=x,y=y)
                    FvAnnotateAdd(x=x,y=y,caption=caption,position='left top',type='max',color=annotcolor)
                    labels_x.append((x-xfirst)/(xlast-xfirst))
                elif word=='min': 
                    x=serOut.idxmin()
                    y=serOut.loc[x]
                    if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=x,y=y)
                    FvAnnotateAdd(x=x,y=y,caption=caption,position='left bottom',type='min',color=annotcolor)
                    labels_x.append((x-xfirst)/(xlast-xfirst))
                else:       # konkrét x-érték (csak egy-szavas lehet)
                    # print('annotate word:' + word)
                    try:
                        x=datefloat(word)
                        y=servalue(serOut,x,True)
                        #print(y)
                        if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=x,y=y)
                        #print(caption)
                        FvAnnotateAdd(x=x,y=y,caption=caption,type='middle',color=annotcolor)
                        labels_x.append((x-xfirst)/(xlast-xfirst))
                    except: pass
            
            # localmax, localmin
            for word in words:
                if beginwith(word,'localmax'):
                    maxdb=0
                    try: maxdb=int(cutleft(word,'localmax'))
                    except: pass
                    points=serlocalmax(serOut,'max',2,0.1)   # width*0.1 távolság egymástól
                    if len(points)>0:
                        if maxdb>0: points=points[:maxdb]
                        for x,y in points:
                            # nem lehet túl közel a fix annotációkhoz
                            bOk=True
                            for label_x in labels_x:
                                if abs((x-xfirst)/(xlast-xfirst) - label_x) < 0.02:
                                    bOk=False
                                    break
                            if bOk: 
                                if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=x,y=y)
                                FvAnnotateAdd(x=x,y=y,caption=caption,position='left top',type='localmax',color=annotcolor)
                elif beginwith(word,'localmin'):
                    maxdb=0
                    try: maxdb=int(cutleft(word,'localmin'))
                    except: pass
                    points=serlocalmax(serOut,'min',2,0.1)   # width*0.1 távolság egymástól
                    if len(points)>0:
                        if maxdb>0: points=points[:maxdb]
                        for x,y in points:
                            # nem lehet túl közel a fix annotációkhoz
                            bOk=True
                            for label_x in labels_x:
                                if abs((x-xfirst)/(xlast-xfirst) - label_x) < 0.02:
                                    bOk=False
                                    break
                            if bOk: 
                                if pattern: caption=pattern.format(label=label,plttype=plttypelast,x=x,y=y)
                                FvAnnotateAdd(x=x,y=y,caption=caption,position='left bottom',type='localmin',color=annotcolor)


        # Egyedi annotációk
        annotplus=params.get('annotplus')           # dictionary  {'2020.03.10':'Járvány kezdete','2022.02.22':'Járvány vége'}
        if annotplus:
            for x,caption in annotplus.items():
                y=servalue(serOut,x,True)
                if annotcolor==None: annotcolor='darkolivegreen'
                FvAnnotateAdd(x=x,y=y,caption=caption,position='right top',color=annotcolor,type='egyedi')


    config.serplotlastG=serOut.copy()       # a hívóhelyen felhasználható
    return 


def FvLinear(ser,resample='',extend=''):
    # lineáris regresszió  (legkisebb négyzetek módszere)
    # Alapesetben a kimeneti ser x értékei meegyeznek az input x értékeivel (nincs resample)

    # resample:    ha nem üres és nagyobb 0-nál, akkor egyenletes x-eloszlás az xmin és xmax között (2 esetén csak a két végpont)
    # extend:   hány ponttal bővítse ki a kemeneti görbét a bal illetve a jobb szélen  (előrejelzésre alkalmazható)
    #     Csak resample>0 esetén érvényesül
    #     '0,20':   bal oldalon nincs bővítés, jobb oldalon 200 pont (a unit a resample-ből adódik)


    serL=ser.dropna()       # üres értéket tartalmazó rekordok kidobása (nem tud velük mit kezdeni a spline)
    serL=serL.sort_index()
    serL=serL.drop_duplicates()   # hibát okozna 
    
    aX=serL.index.array 
    aY=serL.array
    
    datetimeindex = ('timestamp' in str(type(aX[0])))
    if datetimeindex: aX_float=datefloatA(aX)
    else: aX_float=aX

    # Illesztés
    reg = linear_model.LinearRegression()
    
    X=[[aX_float[i]] for i in range(len(aX_float))]     # elemenként kell egy-egy egydimenziós tömb (pl  [[1],[5],[7]])
    print(X)
    reg.fit(X,aY)       
    
    # Koefficiensek, pontosság
    print('coeff:' + str(reg.coef_) + ' y0:' + str(reg.intercept_) + ' score:' + str(reg.score(X,aY)) )
    # - meredekség (feature-önként egy-egy meredekség;  súlyfaktornak is tekinthető)
    # - y0
    # - illesztés minősége

    if resample==None or resample=='' or resample=='0':
        aY=reg.predict(X)      # eredeti x helyeken
        #if datetimeindex:
        #    for i in range(len(aX)): aX[i]=pd.to_datetime(aX[i],unit='D')
        return pd.Series(aY,aX)
    
    else:
        aL=resample.split(',')
        if len(aL)==0: resample='2'
        else: resample=int(aL[0])

        extend_left=0
        extend_right=0
        if extend:
            aL=extend.split(',')
            if len(aL)==1:
                extend_left=int(aL[0])
                extend_right=int(aL[0])
            elif len(aL)>=2:
                extend_left=int(aL[0])
                extend_right=int(aL[1])
        minx=min(aX_float)
        maxx=max(aX_float)
        unit=(maxx-minx)/resample
        minx -= extend_left*unit
        maxx += extend_right*unit
        resample += extend_left + extend_right

        aX=np.linspace(minx,maxx, num=resample, endpoint=True)
        X=[[aX[i]] for i in range(len(aX))]     # elemenként kell egy-egy egydimenziós tömb (pl  [[1],[5],[7]])
        aY=reg.predict(X)

        if datetimeindex:
            for i in range(len(aX)): aX[i]=pd.to_datetime(aX[i],unit='D')
        return pd.Series(aY,aX)


def FvGaussAvg(ser,windowwidth=None,min_periods=None,std=None,trendpoint=None):
    # Gauss súlyozású mozgóátlag  (középre centrált számolással)
    # ser:  tbl is lehet (DataFrame). Tbl esetén az összes oszlopra kiterjed (pl. minden országra)
    # windowidth: a mozgóablak szélessége (hány x-pont). Általában a teljes x-tartomány tizede (hangolásra lehet szükség, a
    #    konkrét igényektől függ a skálázás). 
    # min_periods:  a széleken meddig csökkenhet a mozgóablak szélessége. Default: a windowwidth fele (centrális mozgóátlagolás esetén
    #    legfeljebb ekkora csökkenés engedélyezésére lehet szükség, tovább csökkentve nem nem változik az eredmény).  
    #    Ha megengedett a széleken a tartomány csökkenése, akkor a széleken "biztonságra játszó" számolás érvényesül (kevesebb
    #    értékre átlagol). Egy emelkedő egyenes végén ez egy kisebb meredekségű szakaszt eredményez (tehát a trendet nem teljesen jól jelzi)
    #    Ha a min_periods nagyobb a mozgóablak felénél, akkor a széleken rövidülés.
    # std:  (szigma) Az ablakon belüli gauss súlyozás "csúcsossága".  Default: szigma = windowwidth/4
    # trendpoint:  megadható egy (x,y) trendpont  (általában a jobb szélen túl). Részt vesz az átlagolásban, de a kimenetben nem lesz benne.
    
    # A belső nan értékek kitöltése lineáris interpolációval (nan értékekkel nem működik a mozgóátlag)
    # - a pandas interpolate() függvénye a jobb szélen lévő NaN értékeket is hajlamos kitölteni (az utolsó értékkel)
    #serL=ser.interpolate(limit_area='inside')
    #print('hahó2')
    
    serL=ser
    if not 'DataFrame' in str(type(serL)):         # DataFrame esetén üres táblázatot eredményezne (???)
        serL=serL.dropna()


    if not windowwidth: windowwidth=int(round(len(ser)/10))
    if windowwidth<1: windowwidth=1
    if not min_periods: min_periods=int(windowwidth/2)
    if not std: std=windowwidth/4   
    # print('windowwidth:' + str(windowwidth) + '   std:' + str(std))
    if trendpoint: serL[datefloat(trendpoint[0])]=trendpoint[1]
    #print('trendpoint:' + str(trendpoint) + '\nElőtte: ' + str(serL))
    serL=serL.rolling(windowwidth, min_periods=min_periods, center=True, win_type='gaussian').mean(std=std)
    #print('Utána: ' + str(serL))
    if trendpoint: serL=serL.drop(index=trendpoint[0])
    return serL

def FvGaussStdAvg(ser,windowwidth:int,min_periods=None):
    # Szórás számítása Gauss súlyozású mozgóátlaggal (standard deviation)
    # ser:  tbl is lehet (DataFrame). Tbl esetén az összes oszlopra kiterjed (pl. minden országra)
    if not min_periods: min_periods=int(round(windowwidth/2))
    return ser.rolling(windowwidth, min_periods=min_periods, center=True, win_type='gaussian').std(std=windowwidth/4)

def FvSmoothSpline(ser,diffByMaxmin=0.01,resample='100,500',extend='',trendpoint=None):
    # Elsőként egy lineráris interpoláció, majd spline illesztés
    # ser:  tetszőleges mintavételsor,   az input x értékekeknek nem kell egyenletesnek lenniük és rendezés sem szükséges
    #    - az nan értékeket nem veszi figyelembe
    #    - ha ismétlődő x értékek fordulnak elő, akkor csak egyet őriz meg közülük
    #    - a kimeneti x értékek egyenletesek és rendezettek lesznek
    # diffByMaxmin:  milyen arányú átlagos eltérés megengedett a függvény min-max intervallumához képest
    #     Addig szaporítja a csomópontok számát, amíg a pontosság megfelelővé válik 
    # resample:    egyenletes x-eloszlás az xmin és xmax között
    #     '0,500':     nincs lineáris resample, a spline resample 500 pontos 
    #     '100,500':   100 pontos lineáris resample, majd 500 pontos spline resample
    #     '200':   megegyező lineáris és out resample
    # extend:   hány ponttal bővítse ki a kemeneti görbét a bal illetve a jobb szélen  (ritkán alkalmazható, a spline előrejelzésre nem nagyon alkalmas)
    #     '0,200':   bal oldalon nincs bővítés, jobb oldalon 200 pont


    serL=ser.dropna()       # üres értéket tartalmazó rekordok kidobása (nem tud velük mit kezdeni a spline)
    serL=serL.sort_index()
    serL=serL.drop_duplicates()   # hibát okozna a lineráris és a spline illesztésnél is
    
    std=serL.std()   # nagyjából a max-min felének felel meg (extrém értékektől eltekintve)

    aX=serL.index.array 
    aY=serL.array

    if diffByMaxmin==None: diffByMaxmin=0.01
    s=2*std*diffByMaxmin
    s=(s**2)*len(aX)       # a UnivariateSpline argumentumaként az eltérés-négyzetek összegének max értékét kell megadni

    if resample==None or resample=='': resample='100,500'
    if type(resample)==int:
        resample_linear=resample
        resample_spline=resample
    elif type(resample)==str:
        aL=resample.split(',')
        if len(aL)==1:
            resample_linear=int(aL[0])
            resample_spline=resample_linear
        elif len(aL)>=2:
            resample_linear=int(aL[0])
            resample_spline=int(aL[1])


    datetimeindex = ('timestamp' in str(type(aX[0])))

    if datetimeindex:
        aX_float=[0]*len(aX)
        for i in range(len(aX)): aX_float[i]=aX[i].timestamp()/(24*60*60)
    else: aX_float=aX

    # Elsőként egy lineáris interpoláció, hogy egyenletes eloszlásúak legyenek az x pontok
    #    Ha nem egyenletes, akkor a ritkább régiókban rosszul viselkedhet a spline (erőteljes "kihasasodások" fordulhatnak elő)    
    if resample_linear>0:
        f=interp1d(aX_float,aY,assume_sorted=True)
        lenL=max(aX_float)-min(aX_float)
        minL=min(aX_float)
        maxL=max(aX_float)
        #minL=min(aX_float) + (lenL/resample_linear)/2       # nem megy teljesen a széléig (a széleken bizonytalanság lehet)
        #maxL=max(aX_float) - (lenL/resample_linear)/2
        aX_floatL=np.linspace(minL,maxL, num=resample_linear, endpoint=True)
        aYL=f(aX_floatL)
    else: 
        aX_floatL=aX_float
        aYL=aY

        #if datetimeindex:
        #    aX=[0]*len(aX_floatL)
        #    for i in range(len(aX)): aX[i]=pd.to_datetime(aX_floatL[i],unit='D')
        #else: aX=aX_floatL
        #plt.scatter(aX,f(aX_floatL))
    
    if trendpoint:
        print(trendpoint)
        aX_floatL=np.append(aX_floatL,datefloat(trendpoint[0]))
        aYL=np.append(aYL,trendpoint[1])


    f=UnivariateSpline(aX_floatL,aYL,k=3,s=s)    
    
    # spline resample
    extend_left=0
    extend_right=0
    if extend:
        aL=extend.split(',')
        if len(aL)==1:
            extend_left=int(aL[0])
            extend_right=int(aL[0])
        elif len(aL)>=2:
            extend_left=int(aL[0])
            extend_right=int(aL[1])
    minx=min(aX_float)
    maxx=max(aX_float)
    unit=(maxx-minx)/resample_spline
    minx -= extend_left*unit
    maxx += extend_right*unit
    resample_spline += extend_left + extend_right

    aX_float=np.linspace(minx,maxx, num=resample_spline, endpoint=True)
    aY=f(aX_float)

    if datetimeindex:
        aX=[0]*len(aX_float)
        for i in range(len(aX)): aX[i]=pd.to_datetime(aX_float[i],unit='D')
    else: aX=aX_float
    serout=pd.Series(aY,aX)
    
    return serout

def FvGradient(ser):
    return pd.Series(np.gradient(ser.to_numpy()),ser.index)


# TESZT
if False:
    #aRec=[(1.1,3.4),(2,4.2),(6,-5),(10,-10),(12,1.4)]
    #ser=SerFromRecords(aRec,xname='id')
    #FvPlot(ser,'scatter gauss')
    #plt.show()

    #FvPlot(tbl_excessmortality0['Hungary'],'scatter spline gauss interpolated',gausswidth=20,resample='0,500',splinediff=0.05)
    #plt.show()


    #ser.index
    ''





def FvLinearResample(ser,density=4,count=None,kind='linear'):
    # Egyenletesen elosztott x értékek választása a teljes tartományra. Az y értékek lineáris interpolációval
    # Akkor érdemes alkalmazni, ha az x pontok eredeti eloszlása sztochasztikus (pl. nem szisztemaikus mérési adatok, korrelációszámítások)
    # Megfelelően megválasztott pontsűrűséggel a függvénygörbe simítására is alkalmazható
    # density:  az eredeti x-irányú pontsűrűségének hányszorosa legyen a visszaadott görbe pontsűrűség
    # count:  vagylagos a density-vel - hány pontos legyen a resample 
    aRec=list(zip(ser.index,ser))
    aRec=FvLinearResampleX(aRec,density,count,kind)
    return SerFromRecords(aRec)

def FvLinearResampleX(aXy:list,density=4,count=None,kind='linear'):   # -> aRecEqualized
    # Egyenletesen elosztott x értékek választása a teljes tartományra. Az y értékek lineáris interpolációval
    # Akkor érdemes alkalmazni, ha az x pontok eredeti eloszlása sztochasztikus (pl. nem szisztemaikus mérési adatok, korrelációszámítások)
    # Megfelelően megválasztott pontsűrűséggel a függvénygörbe simítására is alkalmazható (messze nem olyan hatékony, mint a spline)
    # density:  az eredeti x-irányú pontsűrűségének hányszorosa legyen a visszaadott görbe pontsűrűsége 
    # count:  vagylagos a density-vel - hány pontos legyen a resample 
    # kind:  'linear' '2': másodrendű spline   '3': harmadrendű spline, ...    (további lehetőségeket lásd: scipy.interpolate.interp1d)
        
    aXy.sort(key = lambda x: x[0])
    aX,aY=unzip(aXy)
    nLen=len(aXy)
    aX=list(aX)         # módosításra lesz szükség, ezért át kell térni list-re (eredetileg tuple)
    
    datetimeindex = ('timestamp' in str(type(aX[0])))
    if datetimeindex:
        aX_float=[0]*len(aX)
        for i in range(len(aX)): aX_float[i]=aX[i].timestamp()/(24*60*60)
    else: aX_float=aX

    xmin=min(aX_float)
    xmax=max(aX_float)

    
    # unicitás biztosítása  (előfeltétele az interpolációnak)
    xlast=None
    for i,x in enumerate(aX_float):
        if xlast and x<=xlast: 
            x=xlast+(xmax-xmin)/(len(aX)*1000000)       # eltolás az átlagos x-távolság milliomod részével
            aX_float[i]=x
        xlast=x
    
    # interpoláció
    f=interp1d(aX_float,aY,assume_sorted=True,kind=kind)
    if density: count=int(len(aX_float)*density)
    else:
        if count==None: count=100
    aX_float=np.linspace(xmin,xmax, num=count, endpoint=True)
    aY=f(aX_float)

    if datetimeindex:
        aX=[0]*len(aX_float)
        for i in range(len(aX)): aX[i]=pd.to_datetime(aX_float[i],unit='D')
    else: aX=aX_float


    return list(zip(aX,aY))         # a zip összekapcsolja a két tömböt (elempárok)

def FvEventCountPerDay(tblEvents,query=None):
    ''' Időponttal (dátummal) rendelkező esemény-rekordok számlálása naponként 
    tblEvents:  datetime index az elvárás (nem kell unique-nak lennie az index-nek, tetszőleges további kiegészítő adatai lehetnek)
    query:  előzetes szűrés az eseményekre   pl.  "Eseménytípus=='reg'"
    return:  ser,  nap-felbontású idősor (a kulcsa unique)
                   az értéke az adott napra eső események száma
    '''
    if query:
        tblEvents=tblEvents.query(query)
    dates=tblEvents.index.array
    ser=pd.Series([1]*len(dates),dates)
    ser=ser.resample('D').count()
    return ser





def FvScatterPlot(maxsize=1000,defaultsize=100,alpha=0.7,cmap='viridis',colorbar={"label":""},colordefault=None,colorminmax=None):
    # maxsize:  mekkora legyen a legnagyobb pont mérete (points^2-ben mérve) 
    # defaultsize:  mekkora legyen a size=None pontok mérete  (points^2-ben mérve)
    # colordefault: color=None pontok színe  (nem float értéket kell megadni).
    #     - None esetén a színciklus szerinti következő szín 
    # colorminmax: (min,max)  Csak akkor kell megadni, ha nem megfelelő az aColor-ban megadott float-tömb min-max értéke
    # cmap:  
    # colorbar:  dict(label,numformat)     Jobb szélen megjelenő colorbar (cmap alkalmazása esetén sem kötelező megjeleníteni)
    
    #print('aScatterG:' + str(config.aScatterG))

    aX,aY,aColor,aSize,aAnnot=unzip(config.aScatterG)
    
    aColor=list(aColor)
    aSize=list(aSize)

    # Ellenőrzés: van-e legalább kettő nem None eleme az aColor-nak
    nColorDb=0
    for i in range(len(aColor)):
        if aColor[i]==None or pd.isna(aColor[i]): aColor[i]=np.nan
        else: nColorDb+=1
    if nColorDb==0: aColor=colordefault

    if cmap and nColorDb>0 and nColorDb<len(aColor):
        cmap=plt.colors.colormap(cmap)
        cmap.set_bad(color=colordefault,alpha=0.5)
    
    vmin=None
    vmax=None
    if colorminmax:   
        vmin=colorminmax[0]
        vmax=colorminmax[1]

    
    sizemax=0
    for i in range(len(aSize)):
        if not aSize[i]: aSize[i]=defaultsize
        if aSize[i]>sizemax: sizemax=aSize[i]
    for i in range(len(aSize)):
        aSize[i]=(aSize[i]/sizemax)*maxsize

    
    #print('aX:' + str(aX))
    #print('aY:' + str(aY))
    #print('aSize:' + str(aSize))
    #print('aColor:' + str(aColor))
    #print('alpha:' + str(alpha))
    #print('vmin:' + str(vmin))
    #print('vmax:' + str(vmax))


    plt.scatter(aX,aY,c=aColor,s=aSize,alpha=alpha,cmap=cmap,plotnonfinite=True,vmin=vmin,vmax=vmax)

    if (nColorDb>0 or colorminmax) and colorbar:
        label,numformat = dget(colorbar,'label,numformat')
        if numformat: format=mpl.ticker.StrMethodFormatter(FvNumformat(numformat,'x'))
        else: format=None
        plt.colorbar(label=label,format=format)
    
    if aAnnot:
        for i in range(len(aAnnot)):
            if aAnnot[i]: FvAnnotateAdd(aX[i],aY[i],aAnnot[i],color=color_darken(aColor[i],0.5))


def FvScatterAdd(x,y,color=None,size=None,annot=None):
    # Egy pont (kör) hozzáadása a későbbi FvScatterPlot() rajzoláshoz
    
    # x,y:  koordináták  (az x dátum (időpont) is lehet, bár scatter diagramoknál float a jellemző)
    # color, size: nem kell normalizálni (a rajzoláskor lesz normalizálva; tetszőleges járulékos float adat)
    # - a size csak pozitív lehet (vagy None) 
    # - ha valamelyik pontnál nincs megadva (None), akkor a FvScatterPlot() híváskor megadott default szín illetve méret érvényesül
    # annot: nem kötelező

    if size and size<=0: return      # csak pozitív érték adható meg (teljesen kimarad a pont)

    config.aScatterG.append((x,y,color,size,annot))


def FvCorrPlot(tbl,format='simple',order=False,**pltparams):
    '''
    format:
      'simple'      seaborn default formázás
      'triangle'    annotálásokkal, alul 10-ig lépcsőzetes annotálások, 10 felett ferde feliratok
    '''
    
    tblcorr=tbl.corr()

    if order:
        tblcorr['corrsum']=tblcorr.sum()
        tblcorr=tblcorr.sort_values(by='corrsum')
        tblcorr=tblcorr[tblcorr.index.array]        # oszlopok átrendezése a sorok sorrendje alapján ('corrsum' oszlop kimarad)

    if format=='simple':
        pltinit(pltparams)
        sb.heatmap(tblcorr, cmap="BrBG_r",vmin=-1, vmax=1)
        pltshow()
    
    elif format=='normal':
        dsetsoft(pltparams,suptitle='Korrelációk',height=0.8,left=0.25,bottom=0.3,right=0.97)
        pltinit(**pltparams)
        numcols=len(list(tblcorr))

        aCaption=tblcorr.index.array
        sb.heatmap(tblcorr, cmap="BrBG_r",vmin=-1, vmax=1)
        plt.xticks(rotation=-35,ha='left',fontsize=8)
        plt.tick_params(axis='x', which='major', pad=1)
        
        pltshow()      # annot_ystack: lépcsőzetes elrendezés 12-es csoportokban        


    elif format=='triangle':
        print('pltparams:' + str(pltparams))
        dsetsoft(pltparams,suptitle='Korrelációk',height=0.8,left=0.25,bottom=0.3,right=0.97)
        print('pltparams:' + str(pltparams))
        pltinit(**pltparams)
        numcols=len(list(tblcorr))

        aCaption=tblcorr.index.array
        mask = np.triu(np.ones_like(tblcorr))
        mask = mask[1:, :-1]
        tblcorr = tblcorr.iloc[1:,:-1].copy()
        sb.heatmap(tblcorr, cmap="BrBG_r",mask=mask,vmin=-1, vmax=1,xticklabels=(numcols>12))
        # anotációk az átlóban
        for i,caption in enumerate(aCaption):
            if i>0 and i<len(aCaption)-1: FvAnnotateTwo((0.5 + i,i),(i,i-0.5),caption,5,8)
        
        # xlabels
        y0=len(aCaption)-1    # a seaborn diagram bal felső sarka az origo
        if numcols<=12:
            # x tengely feliratozása
            for i,caption in enumerate(aCaption):
                FvAnnotateAdd(0.5 + i,y0,caption,'bottom right')    
        else:
            plt.xticks(rotation=-35,ha='left',fontsize=8)
            plt.tick_params(axis='x', which='major', pad=1)
        
        pltshow(annot_fontsize=8,annot_yoffset=10,annot_ystack=(y0,12))      # annot_ystack: lépcsőzetes elrendezés 12-es csoportokban        






def FvCrosscorr(tbl1,country1:str,tbl2,country2:str,par='',datumfirst='',omit=0):      # ->(lag,korr,ser,tblout)
    # tbl1,tbl2:  pl. tbl_cases, tbl_deaths
    # par: "grad":  a gradiensre nézze a keresztkorrelációt
    # datumfirst:   milyen dátumtól
    # omit:  hány napot hagyjon el a szélekről

    # ha lag negatív, akkor a country2 felfutásai általában megelőzik a country1 felfutásait

    smoothhalf=20
    smoothlen=2*smoothhalf

    if not datumfirst:
        datumfirst=tbl1.index.min()+pd.DateOffset(omit)
        # Az első nem NaN record keresése
        for i in range(len(tbl1.index)):
            if not np.isnan(tbl1[country1][i]) and not np.isnan(tbl2[country2][i]):
                if i>smoothhalf: 
                    datumfirst=tbl1.index[i]
                break

    datumlast=tbl1.index.max()-pd.DateOffset(omit)

    ser1=tbl1.loc[datumfirst:datumlast,country1]
    ser2=tbl2.loc[datumfirst:datumlast,country2]       # kipróbáltam negatív előjellel is, de a korreláció (az előjelet leszámítva) ugyanaz lett

    if par=="grad":
        ser1 = pd.Series(FvNormalize(np.gradient(ser1.to_numpy())),ser1.index)
        ser1 = ser1.rolling(smoothlen, min_periods=1, center=True, win_type='gaussian').mean(std=smoothlen/4)
        ser2 = pd.Series(FvNormalize(np.gradient(ser2.to_numpy())),ser2.index)
        ser2 = ser2.rolling(smoothlen, min_periods=1, center=True, win_type='gaussian').mean(std=smoothlen/4)
    
    tblout=pd.DataFrame({
        tbl1.name + '.' + country1:ser1,
        tbl2.name + '.' + country2:ser2
    })

    #print(ser1)
    #print(ser2)

    # Keresztkorreláció  (ccf: cross correlation function)
    forwards=stat.ccf(ser2,ser1,adjusted=False,fft=False)         # jó lenne, ha meg lehetne adni a lag-tartományt, de a ccf nem tud ilyesmit
    backwards=stat.ccf(ser1,ser2,adjusted=False,fft=False)[::-1]      # megfordítás
    backwards=backwards[:-2]        # lag=0 nem kell  (átfedés a forwards-szel)

    ser=pd.concat([pd.Series(backwards,index=range(-len(backwards),0)),pd.Series(forwards,index=range(0,len(forwards)))])
    if country1!=country2 or tbl1.name!=tbl2.name:
        ser=ser.loc[-100:100]        # 100 nap feletti korreláció nem érdekel (kivéve: autokorreláció)

    if par=='onlypositive': ser=ser.loc[0:]

    return (ser.idxmax(),ser.max(),ser,tblout)



# ANNOTATE
def FvAnnotateAdd(x,y,caption,position='right bottom',color=None,type=''):
    # x,y:  az annotálandó pont helye (eredeti koordinátákban; az x általában dátum)
    # caption: felirat  (több soros is lehet \n jelekkel)
    #     példa:  egy görbe jobb szélső pontja    
    #         caption = '... (' + '{:.1f}'.format(ser[-1]*100) + '%)'
    #         FvAnnotateAdd(ser.index[-1],ser[-1],caption)
    #     példa:  egy görbe közbenső pontja   (a FvPlot annotplus argumentumában is megadható)
    #         datumL='2021.02.01'
    #         caption = '... (' + '{:.1f}'.format(ser.loc[datumL]*100) + '%)'
    #         FvAnnotateAdd((pd.Timestamp(datumL),ser.loc[datumL],caption))
    #     példa:  egy görbe maximumpontja
    #         datumL=ser.idxmax()
    #         caption = '... (' + '{:,.0f}'.format(ser.loc[datumL]) + ')'
    #         FvAnnotateAdd((pd.Timestamp(datumL),ser.loc[datumL],caption,15))          # yoffset is meg van adva
    #  color:  a label betűszíne   (pl. színszó, színkód, stb)
    #     - ha nincs megadva, akkor az utoljára rajzolt volna színe (sötétítve)
    #  type:  szöveges típusjelzés.  Péld: a FvPlot a first,last,middle,max,min típusjelzéseket használja
    #     - a FvAnnotatePlot függvény annot_count argumentumában hivatkozni lehet a típusokra
    
    if y==np.nan: return        # not-a-number érték nem adható meg

    if not color:
        lines=plt.gca().get_lines()
        if len(lines)>0: 
            color=lines[-1].get_color()        # így lehet bekérni utólag a vonal színét
            color=color_darken(color,0.5)
    
    #print('FvAnnotateAdd (utána):  caption:' + caption + '  color:' + str(color))

    config.aAnnotG.append((x,y,caption,position,color,type))


def FvAnnotatePlot(xoffset=8,yoffset=5,fontsize=8,heightfactor=1.3,annot_count=None,serbaseline=None,color=None,ymin=None,ymax=None,ystack=None):
    # Diagramok annotálása, az átfedések minimalizálásával
    #   - előfeltétel:  FvAnnotateAdd sorozat
    #   - csak függőleges irányú label-eltolások, a lehető legkisebb mértékben (mind esztétikai, mind funkcionális szempontból
    #       előny, ha a címkék pozícionálása viszonylag egységes)
    #   - nyilakat rajzol a címkéktől az annotált pontig
    #   - függőleges irányban megőrzi az eredeti sorrendet (csak rendkívül erős tömörülés esetén fordulhat elő sorrendcsere)
    #   - ha több rajzterület is van, akkor az aktuális rajzterületre rajzol (rajzterületenként külön annotálás kell)
    # xoffset,yoffset:  a label alapértelmezett pozíciója az annotálandó ponthoz képest, point-ban  (a betűmérethez igazítható)
    #     - mindkét érték legyen pozitív (a position-nak megfelelő előjelváltásokról a függvény gondoskodik)
    #     - kezdőérték, átfedések minimalizálása során eltérhet ettől a tényleges pozíció
    # fontsize:  egyetlen szám (globális beállítás), vagy egy dictionary típusToFontsize hozzárendelésekkel (string-dict formátum is lehet) 
    #     - ha egyetlen szám, akkor a localmax,localmin jelölőkre eggyel kisebb méret érvényesül, minden másra a megadott méret  
    #     - a FvPlot a first,last,middle,max,min,localmax,localmin,egyedi típusneveket alkalmazza;   minden egyéb: "other")
    # heightfactor: milyen szorzóval számolja a sormagasságokat (az eredeti fontmagassághoz képest) 
    # color:  mindegyik címke ezzel a színnel jelenjen meg. 
    #    - ha nincs megadva, akkor a FvAnnotateAdd-ban megadott érték (default: az utoljára rajzolt vonal színe, sötétítve), annak hiányában
    #       a default színciklus következő színe. Lásd még: config.speccolorsG
    # annot_count:  globálisan vagy típusonként megadható, hogy hány annotáció jelenjen meg (az alapvonaltól legtávolabbi jelölők jelennek meg)
    #    - szám, dictionary vagy kóldista-string.    Default:  'first:20//middle:20//last:20//localmax:10//localmin:10//other:20'   
    #    - a FvPlot által alkalmazott típusok:  first,middle,last, localmax,localmin  (más típus is előfordulhat lásd FvAnnotateAdd). Továbbá: 'other'
    #    - a config.linecolorsG-ben megadott label-ekre nem vonatkozik a korlátozás (mindenképpen megjelennek)
    #    - default alapvonal: x tengely  (lásd serbaseline)
    # serbaseline: a annot_count algoritumshoz tartozó alapvonal.  Default: x tengely (y=0 függvény)
    #    - egy series adható meg, ami legalább két x-re tartalmaz y értéket (nem lineáris esetben sok pont kell). Minden más pontra lineáris interpolációval lesz kiszámolva az y érték.
    # ymin,ymax:  egyetlen címke se kerüljön a megadott határvonal alá illetve fölé (csak az egyik érvényesül; nem kötelező megadni)
    # ystack: (y,stacklen)  vagy array of (y,stacklen);   az adott y-hoz tartozó pontok címkéinek lépcsőzetes eltolása stacklen ciklusokban
    #    - példa: x tengely címkézése;  a függőleges címkefeliratok helyett alkalmazható, pl. egy korrelációs táblázatban 
    
    #print('ymax:' + str(ymax)

    if xoffset==None: xoffset=8
    if yoffset==None: yoffset=5

    if fontsize==None: fontsize='localmax,localmin:8//other:8'       # None,  8,  '8',  'last:8//other:7', {'last':8,'other':7}
    elif type(fontsize)==int:
        fontsize={'localmax':fontsize-1,'localmin':fontsize-1,'other':fontsize}
    
    if type(fontsize)==str:
        try: fontsize=int(fontsize)
        except:
            fontsize=kodlistaToDict(fontsize,bIntValue=True)
            if fontsize=={}: fontsize=={'localmax':7,'localmin':7,'other':8}

    if heightfactor==None: heightfactor=1.3
    
    ystep=(8*heightfactor)*0.2     # label léptetések nagysága az átlagos label-magassághoz képest (fontsize=8 értékkel számol)

    if annot_count==None: annot_count='first,middle,last:20//localmax,localmin:10//other:20'
    # string formátum esetén konverzió integer-re vagy dict-re
    if type(annot_count)==str:
        try: annot_count=int(annot_count)
        except:
            annot_count=kodlistaToDict(annot_count,bIntValue=True)
            if annot_count=={}: annot_count=25          # ha rossz volt a formátum, akkor 25 darabos globáli limit


    maxcycles=200

    if len(config.aAnnotG)==0: return

    #aRectFix=[]
    #xlim=plt.xlim()
    #xlim=axeskoord(xlim[0],xlim[1],'point');
    #if ymax!=None: 
        
    #    aRectFix.append((xlim[0],ymax,xlim[1],ymax+1000))
    #if ymin!=None: aRectFix.append((xlim[0],ymin-1000,xlim[1],ymin))


    ax=plt.gca()
    dpi=plt.rcParams['figure.dpi']


    # Csoportosítani kell az axindex mező alapján (rajzterületenként külön ciklusok)
    #config.aAnnotG.sort(key = lambda x: x[5])        
    #for axindex,itersub in groupby(config.aAnnotG,lambda x: x[5]):
    #    # A rajzterület aktuálissá tétele (a plt.gca() ezt a rajzterületet adja vissza)
    #    plt.sca(plt.gcf().axes[axindex])
    #    aAnnotsub=list(itersub)


    aAnnot=config.aAnnotG.copy()
    nCount=len(aAnnot)
    aX,aY,aCaption,aPosition,aColor,aType = unzip(aAnnot)

    # FELESLEGES ANNOTÁCIÓK ELHAGYÁSA  (darabszám limit feletti, localmax esetén a baseline alatti, localmin esetén a baseline feletti)
    if annot_count:
        # Eltérések a baseline-hoz képest
        # minden x értékhez kell egy yBaseline (aYBase)
        if serbaseline==None: aYBase=[0]*len(aX)
        else: aYBase=servalueA(serbaseline,aX)

        aYDiff=[0]*len(aY)
        ydiffmax=0      # a legnagyobb eltérés a baseline-tól (abszolút érték)
        for i in range(len(aY)): 
            aYDiff[i]=abs(aY[i]-aYBase[i])
            if aYDiff[i]>ydiffmax: ydiffmax=aYDiff[i]

        sortarrays3(aYDiff,aAnnot,aYBase,True)      # csökkenő sorrend, aAnnot és aYBase szinkronizált rendezése

        if type(annot_count)==int:
            aAnnot=aAnnot[:annot_count]

        elif type(annot_count)==dict:
            # számlálók létrehozása típusonként
            annot_counter=annot_count.copy()
            for typeL in annot_counter: annot_counter[typeL]=0    # számlálók nullázása
        
            aAnnot_=[]
            for i in range(len(aAnnot)):
                annot=aAnnot[i]
                typeL=annot[5]
                y=annot[1]
                
                if typeL=='localmax':
                    if y<=aYBase[i]: continue                # csak a baseline feletti lokális maximumok kellenek
                    if aYDiff[i]<=ydiffmax*0.05: continue    # abszolút eltérés nem lehet túl kicsi
                elif typeL=='localmin':
                    if y>=aYBase[i]: continue                # csak a baseline alatti lokális minimumok kellenek
                    if aYDiff[i]<=ydiffmax*0.05: continue    # abszolút eltérés nem lehet túl kicsi

                if typeL==None or typeL=='': typeL='other'
                nDb=annot_counter.get(typeL)
                # ha nincs ilyen típus, akkor 'other'
                if nDb==None and typeL!='other': 
                    typeL='other'
                    nDb=annot_counter.get('other')
                # Ha nincs limit az adott típusra, akkor mindegyik record megtartandó
                if nDb==None: aAnnot_.append(annot)
                # Ha van darabszám-limit, akkor ellenőrzés
                else:
                    nDbMax=annot_count.get(typeL)
                    if nDbMax and nDb<nDbMax:
                        annot_counter[typeL]=nDb+1
                        aAnnot_.append(annot)
            aAnnot=aAnnot_
            
        # Tömbök incializálása (változhatott az aAnnot)
        nCount=len(aAnnot)
        aX,aY,aCaption,aPosition,aColor,aType = unzip(aAnnot)

    

    aX_points,aY_points = axeskoordA(aX,aY,'point')
    #print('aX_points:' + str(aX_points))

    def sub_fontsize(labeltype,color):
        if type(fontsize)==int: result=fontsize             # fontsize: input argumentum (int vagy dict lehet)
        elif type(fontsize)==dict:
            try: result=int(fontsize[labeltype])
            except:
                try: result=int(fontsize['other'])
                except: result=8
        if color=='kiemelt': result=result+1
        return result


    def labelsize(caption,labeltype,color):
        aSor=caption.splitlines()
        height=0
        width=0
        for sor in aSor:
            w,h,d = mpl.textpath.TextToPath().get_text_width_height_descent(sor, 
                                mpl.font_manager.FontProperties(family="Century Gothic", size=sub_fontsize(labeltype,color)), False)
            height += h*heightfactor     # nem kell a d ("descent"), mert már benne van a h-ban
            if w>width: width=w
        return width,height



    # Induló aRect-ek:
    aRect=[None]*nCount
    aYLabel0=[0]*nCount          # a címkék bal alsó sarkának kiinduló pozíciója (point-ban)
    for i in range(nCount):
        caption=aCaption[i]
        labeltype=aType[i]
        width,height = labelsize(caption,labeltype,aColor[i])
        xoffsetL=xoffset
        yoffsetL=-yoffset - height/2      # default: bottom;   a height/2 azért kell, mert a nyíl középről indul
        #print('yoffsetL:' + str(yoffsetL))
        position=aPosition[i]
        if not position: position='right bottom'
        words=position.lower().split()
        if 'left' in words: 
            xoffsetL=-xoffsetL-width        # jobb szélről indul a nyíl
        if 'top' in words: 
            yoffsetL=-yoffsetL 
        aRect[i]=(aX_points[i]+xoffsetL,aY_points[i]+yoffsetL,aX_points[i]+xoffsetL+width,aY_points[i]+yoffsetL+height)
    
    
    # ystack: az azonos y értékhez tartozó címkék lépcsőzetes eltolása
    if ystack:
        if type(ystack)==tuple: aYStack=[ystack]     # tuple vagy arra of tuple az input
        else: aYStack=ystack
        # print('aYStack:' + str(aYStack))
        for ystack in aYStack:        
            y=ystack[0]                 # (y,stacklen);   az y a címkézendő pontsorozat y koordinátája
            stacklen=ystack[1]
            # Az adott y-hoz tartozó rect-ek kiválasztása
            aStack=[]
            for i in range(len(aY)):
                if aY[i]!=y: continue
                aStack.append((aX[i],i))          # az index is tárolandó
            if len(aStack)>1:               # ha tartozik legalább két címke a megadott y-hoz
                sortrecords(aStack,0)       # nem volt elvárás, hogy az input rendezett legyen x irányban
                width,height=labelsize('Sample')       
                nShift=stacklen-1    # induló eltolás
                if len(aStack)<stacklen: nShift=len(aStack)-1    
                for i in range(len(aStack)):
                    index=aStack[i][1]
                    shift=height*nShift
                    aRect[index]=FvRectShiftY(aRect[index],-height*nShift)     # lefelé tolás, lépcsőzetesen
                    nShift-=1
                    if nShift<0: nShift=stacklen-1


    # aRect-ek léptetése
    aYLabel0=[0]*nCount                 # Az induló y-értékek tárolás
    for i in range(nCount): aYLabel0[i]=aRect[i][1]         # a rect-ek alsó határa
    
    cycle=0
    steps=''
    aStepsize=[ystep,ystep/2,ystep/4,ystep/8,ystep/16,ystep/32,ystep/64]
    for stepsize in aStepsize:
        #print('Stepsize: ' + str(stepsize))
        for cycleL in range(maxcycles):
            # print('cycle:' + str(cycle))
        
            nShiftCount=0
            stepsL=['0']*nCount
            # Alciklus a címkékre
            for i in range(nCount):
                y0=aYLabel0[i]          # az eredeti pozíció point-ban
                # kísérlet felfelé illetve lefelé tolásra
                rect=aRect[i]
                aOverlap=[0]*3
                for dir in range(3):    # 0 jelenlegi helyzet,  1: eltolás felfelé  2: eltolás lefelé
                    if dir==0: rectL=rect
                    elif dir==1: rectL=FvRectShiftY(rect,stepsize)
                    elif dir==2: rectL=FvRectShiftY(rect,-stepsize)
                    # Átfedés a többi címkével
                    for j in range(nCount):
                        if j==i: continue
                        overlap=FvRectOverlapHeight(rectL,aRect[j])
                        #if overlap>aOverlap[dir]: aOverlap[dir]=overlap
                        aOverlap[dir]+=overlap*(1 - (abs(aRect[j][1]-y0)/10000))    # minél messzebb van a másik az eredeti pozíciótól, annál kisebb az átfedés súlya
                        # if cycle==8 and i==7 and overlap>0: 
                        #    print('j:' + str(j) + ' overlap:' + str(overlap) + '  ' + str(rectL) + '  ' + str(aRect[j]) )
                    # Átfedés a fix rect-ekkel
                    #for rect in aRextFix:
                    #    overlap=FvRectOverlapHeight(rectL,rect)
                    #    aOverlap[dir]+=overlap
                    # if dir==0 and aOverlap[0]==0: break     # ha jelenleg nincs átfedés, akkor nem kell megnézni a két eltolási lehetőséget
                    #elif dir==1 and ymax!=None and rectL[3]>ymax: aOverlap[dir]+=rectL[3]-ymax
                    #elif dir==2 and ymin!=None and rectL[1]<ymin: 
                    #    aOverlap[dir]+=ymin-rectL[1]
                    #    print('hahó')
            
                # Javul-e a helyzet valamelyik eltolással
                ystepL=0
                # Ha nincs semmilyen átfedés 
                if aOverlap[0]==0:
                    if stepsize==ystep/64:     # az utolsó körben egy záró korrekció
                        # Ha a címke lejjebb van az eredeti pozíciónál, és felfelé léptetéssel sem lesz semmilyen átfedés, akkor felfelé léptetés
                        if aRect[i][1]+stepsize<aYLabel0[i] and aOverlap[1]==0: ystepL=stepsize
                        # Ha a címke feljebb van az eredeti pozíciónál, és lefelé léptetéssel sem lesz semmilyen átfedés, akkor lefelé léptetés
                        elif aRect[i][1]-stepsize>aYLabel0[i] and aOverlap[2]==0: ystepL=-stepsize

                # Ha a címke felül van, akkor a felfelé tolás preferált
                elif aY_points[i]<=(aRect[i][1]+aRect[i][3])/2:
                    if aOverlap[1]<=aOverlap[0]: ystepL=stepsize       
                    elif aOverlap[2]<aOverlap[0]: ystepL=-stepsize          #  and rect[1]-stepsize>ymin
                # Ha a címke alul van, akkor a lefelé tolás preferált
                else:
                    if aOverlap[2]<=aOverlap[0]: ystepL=-stepsize       
                    elif aOverlap[1]<aOverlap[0]: ystepL=stepsize   


                # Az eltolás végrehajtása
                #if cycle in [0,1]:
                #    print('cycle:' + str(cycle) + ' i:' + str(i) + '  step:' + str(ystepL) + '  rect_now:' + str(aRect[i]))
                #    print('overlap:' + str(aOverlap[0]) + '  fel:' + str(aOverlap[1]) + '  le:' + str(aOverlap[2]) + '\n')
                if ystepL!=0:
                    aRect[i]=FvRectShiftY(rect,ystepL)
                    nShiftCount+=1
                    if ystepL>0: stepsL[i]='F'      # bejegyzés az eltolás táblázatba (step-log)
                    else: stepsL[i]='L'

            steps=steps + '\n' + ''.join(stepsL) + '   step: ' + str(stepsize)
            cycle+=1
            if nShiftCount==0: break
   
    
    # Címkék felső vagy alsó határa
    #print('ymax:' + str(ymax))
    if ymax!=None:
        y2=unzip(aRect,3)
        diff=max(y2)-ymax
        if diff>0: 
            for i in range(len(aRect)): aRect[i]=FvRectShiftY(aRect[i],-diff)
    elif ymin!=None:
        y1=unzip(aRect,1)
        diff=ymin - min(y1)
        if diff>0: 
            for i in range(len(aRect)): aRect[i]=FvRectShiftY(aRect[i],diff)

    for i in range(nCount):
        # Teszt: rect kirajzolása  (nem jó, adat-koordináták kellenének)
        #rect = patches.Rectangle((aRect[i][0], aRect[i][1]), aRect[i][2]-aRect[i][0], aRect[i][3]-aRect[i][1], linewidth=1, edgecolor='r', facecolor='none')
        #plt.gca().add_patch(rect)
        
        #if 'Hungary' in aCaption[i]: print('Hungary left, right:' + str(aRect[i][0]) + ' ' + str(aRect[i][2]))
        #elif 'Romania' in aCaption[i]: print('Romania left, right:' + str(aRect[i][0]) + ' ' + str(aRect[i][2]))

        
        rad=0.2         # nyilak görbületének erőssége
        if aY_points[i]>(aRect[i][1]+aRect[i][3])/2: rad=-rad    # alulról induló nyíl esetén negatív a görbület
        # angleB=math.atan(yoffsetL/xoffset)*(180/math.pi)
        # arrowprops=dict(arrowstyle='-|>',shrinkA=0,shrinkB=0,mutation_scale=6,connectionstyle='angle3,angleA=0,angleB=' + str(angleB)))
        
        if aRect[i][2]<aX_points[i]: xoffset=aRect[i][2]-aX_points[i]       # negatív xoffset, haling=right
        else: xoffset=aRect[i][0]-aX_points[i]           # alapesetben pozitív xoffset (halign=left), Ha negatív, akkor halign=right

        yoffsetL=((aRect[i][1]+aRect[i][3])/2) - aY_points[i]    # a nyíl a bal vagy a jobb szél közepéről indul 

        caption=aCaption[i]
        labeltype=aType[i]
        
        fontsizeL=sub_fontsize(labeltype,aColor[i])

        colorL=None
        if aColor[i]=='kiemelt': colorL='red'
        else:
            # annotcolorG
            words=caption.split()   
            if len(words)>0:                        # kiemelt felirat színe (a felirat első szava alapján)
                firstword=words[0]
                colorL=dget(config.annotcolorG,firstword)
            if colorL==None: colorL=aColor[i]       # FvAnnotateAdd-ben megadott egyedi szín
            if colorL==None: colorL=color           # argumentumként megadott közös szín
            if colorL==None: colorL='black'         # default: fekete
        
        FvAnnotateOne(aX[i],aY[i],caption,xoffset=xoffset,yoffset=yoffsetL,fontsize=fontsizeL,color=colorL)
    
    config.aAnnotG=[]
    # print('Annotation cycle:' + str(cycle) + steps)       # debug


def FvAnnotateBarH(fontsize=8,gap=5,numformat='2f'):
    # Horizontális oszlopdiagram annotálása
    #   gap:  point-ban
    #   decimal:  hány tizedesjegy jelenjen meg 
    # Kiindulópont:  https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    
    rects = plt.gca().patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Vertical alignment for positive values
        ha = 'left'
        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            gap *= -1
            ha = 'right'

        formatstring=FvNumformat(numformat)
        label = formatstring.format(x_value)

        plt.annotate(label,(x_value, y_value),     
            xytext=(gap, 0), textcoords="offset points", 
            va='center', ha=ha, fontsize=fontsize)

def FvAnnotateOne(x,y,caption,xoffset=10,yoffset=10,fontsize=8,color='black',alpha=0.9,alphabbox=0.3):
    # Címke és nyíl megjelenítése
    # Ha xoffset<0, akkor a címke jobbra igazított és a nyíl a jobb szélről indul, egyébként balra igazított és a nyíl bal szélről indul
    # y irányban a nyíl a bal vagy a jobb szél közepéről indul
    
    rad=0.2         # nyilak görbületének erőssége
    if yoffset<=0: rad=-rad    # alulról induló nyíl esetén negatív a görbület

    relpos=(0,0.5)
    ha='left'

    if xoffset<0:
        rad=-rad
        relpos=(1,0.5)
        ha='right'
    
    #print('caption:' + caption + ', x:' + str(x) + ', y:' + str(y) + ' xoffset:' + str(xoffset) + ' yoffset:' + str(yoffset))
    plt.annotate(caption,(x,y),xycoords='data',
                    xytext=(xoffset,yoffset), textcoords='offset points',
                    ha=ha,va='center',fontsize=fontsize, alpha=alpha, color=color,
                    bbox=dict(visible=True,edgecolor=None,color='white',alpha=alphabbox,pad=0.1),
                    arrowprops=dict(arrowstyle='-|>',shrinkA=0,shrinkB=0,relpos=relpos,alpha=0.4,
                                    mutation_scale=fontsize-2,connectionstyle='arc3,rad=' + str(rad)))

def FvAnnotateTwo(xy1,xy2,caption,xoffset=10,yoffset=10,leftright='right',fontsize=8,color='black',alpha=0.9,alphabbox=0.3):
    # egy címke két ponthoz.  Példa: két érték különbségének vagy hányadosának kiírásakor
    # - az xoffset a jobbra eső ponthoz képest értendő (leftright='left' esetén a balra eső ponthoz képest)
    # - az yoffset az első pont x értékéhez képest értendő
    
    xy1_point=axeskoord(xy1[0],xy1[1],'point')
    xy2_point=axeskoord(xy2[0],xy2[1],'point')

    if leftright=='right':
        if xy2_point[0]>xy1_point[0]: 
            xoffset1=xoffset + (xy2_point[0]-xy1_point[0])
            xoffset2=xoffset
        else: 
            xoffset1=xoffset
            xoffset2=xoffset + (xy1_point[0]-xy2_point[0])
    else:
        if xy2_point[0]<xy1_point[0]: 
            xoffset1=xoffset + (xy1_point[0]-xy2_point[0])
            xoffset2=xoffset
        else: 
            xoffset1=xoffset
            xoffset2=xoffset + (xy2_point[0]-xy1_point[0])
    
    yoffset1=yoffset
    yoffset2=(xy1_point[1] + yoffset) - xy2_point[1] 

    FvAnnotateOne(xy1[0],xy1[1],caption=caption,xoffset=xoffset1,yoffset=yoffset1,fontsize=fontsize,
                        color=color,alpha=alpha,alphabbox=alphabbox)
    FvAnnotateOne(xy2[0],xy2[1],caption=caption,xoffset=xoffset2,yoffset=yoffset2,fontsize=fontsize,
                        color=color,alpha=0,alphabbox=0)




# RECT
def FvRectOverlap(rectA:tuple,rectB:tuple):  #  ->float  (az átfedő terület nagysága)
    # rect[0]:x_left   rect[1]:y_bottom   rect[2]:x_right   rect[3]:y_top
    if rectA[2]<=rectB[0] or rectA[0]>=rectB[2] or rectA[3]<=rectB[1] or rectA[1]>=rectB[3]: return 0
    x = max(rectA[0], rectB[0])         # nagyobbik left
    y = max(rectA[1], rectB[1])         # nagyobbik bottom
    w = min(rectA[2], rectB[2]) - x     # kisebbik right - nagyobbik left
    h = min(rectA[3], rectB[3]) - y     # kisebbik top - nagyobbik bottom
    return (w*h)   

def FvRectOverlapHeight(rectA:tuple,rectB:tuple):  #  ->float  (az átfedés magassága; a szélessége érdektelen)
    if rectA[2]<=rectB[0] or rectA[0]>=rectB[2] or rectA[3]<=rectB[1] or rectA[1]>=rectB[3]: return 0
    return min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])

def FvRectArea(rect:tuple): # ->float     (a rect területe)
    return (rect[2]-rect[0])*(rect[3]-rect[1])

def FvRectShiftY(rect:tuple,deltay:float):  # ->tuple   
    return (rect[0],rect[1]+deltay,rect[2],rect[3]+deltay)



        
# TESZT
if False:
#aAnnot=[(1,2,'első',None),(1.05,2,'Második',None),(1.1,2,'Harmadik',None),(1.15,2.01,'Negyedik',None),
#        (1.2,1.99,'ötödik',None),(1.25,1.98,'Hatodik',None),(1.3,2.02,'Hetedik',None),(1.35,2,'Nyolcadik',None),
#                (3,3,'Utolsó',None)]
#aX,aY,aCaption,*aOther=unzip(aAnnot)
#plt.plot(aX,aY)
#FvAnnotatePlot(ystep=2,xoffset=10,fontsize=8,maxcycles=100,heightfactor=1.3)
#plt.show()
            
                
#sor='Második'
#fontname='Century Gothic'     # 'Arial'
#mpl.textpath.TextToPath().get_text_width_height_descent(sor, 
#                                mpl.font_manager.FontProperties(family=fontname, size=9), False)
    end


# KOORDINÁTA TRANSZFORMÁCIÓK
def datefloat(x): 
    # x:  ha int vagy float (akár stringként is), akkor nincs konverzió
    #     ha string (de nem értelmezhető számnak), akkor megpróbálja dátumként értelmezni és unix float-ra konvertál
    #     ha timestamp, akkor unix-float ra konvertál
    #     egyébként megpróbálja a közvetlen float konverziót
    #  lehet string (pl. '2021.01.02','20210102')   vagy pd.Timestamp is
    if type(x) in [float,int]: return float(x)
    elif type(x)==str: 
        if len(x)!=10:      # elsőként megnézi, hogy számként értelmezhető stringről van-e szó (kivéve: 20220101)
            try: return float(x)      
            except: pass
        return pd.to_datetime(x).timestamp()/(24*60*60)       
    elif 'Timestamp' in str(type(x)): return x.timestamp()/(24*60*60)     # érthetetlen, hogy miért nem tudja automatikusan a pandas ezt a konverziót
    else: return float(x)

def datefloatA(aX): 
    # date: lehet string (pl. '2021.01.02','20210102')   vagy pd.Timestamp is
    aOut=[0]*len(aX)
    for i in range(len(aX)): aOut[i]=datefloat(aX[i])
    return aOut

def axeskoord(x,y,  unit='0-1'):                # Data to axeskoord   (axeskoord:  a rajzolási tartomány határai [0,1]-nek felelnek meg)
    # a diagram rajzolási területéhez viszonyított relatív koordinátákat adja vissza,    [0-1] vagy inch vagy point mértékegységben
    # Fontos: a diagram-terület átméretezésével érvénytelenné válhatnak az inch-ben és point-ban visszaadott koordináták
    # x: dátum is lehet  (szöveges vagy pd.Timestamp)
    # A speciális skálázásokat is tudja kezelni  (pl. log)
    # unit:  '0-1', 'point', 'inch' (=72 point)
    x=datefloat(x)
    # koordináták [0-1] mértékegységgel  (axespercent)
    xy=plt.gca().transLimits.transform(plt.gca().transScale.transform((x,y)))    # figyelembe veszi a log skálázást is
    xunit,yunit=axesunits(unit)       # unit='0-1' esetén (1,1) a result
    return (xy[0]*xunit,xy[1]*yunit)

def axeskoordA(aX,aY,unit='0-1'):
    # a diagram rajzolási területéhez viszonyított relatív koordinátákat adja vissza [0-1], inch vagy point mértékegységben
    # Fontos: a diagram-terület átméretezésével érvénytelenné válhatnak az inch-ben és point-ban visszaadott koordináták
    # aX: dátumokat is tartalmazhat  (szöveges vagy pd.Timestamp)
    # A speciális skálázásokat is tudja kezelni  (pl. log)
    # unit:  '0-1', 'point', 'inch'
    xunit,yunit=axesunits(unit)       # '0-1' esetén (1,1) a result;   a bbox teljes szélessége/magassága point-ban / inch-ben
    #print('xunit:' + str(xunit))
    xlim=plt.xlim()         # dummy hívás;  enélkül a transform műveletek nem működnek jól (vélhetően inicializál valamit)
    aXOut=[0]*len(aX)
    aYOut=[0]*len(aY)
    for i in range(len(aX)):
        x=datefloat(aX[i])
        xy=plt.gca().transLimits.transform(plt.gca().transScale.transform([x,aY[i]]))    # figyelembe veszi a log skálázást is
        # - elvileg mindkét koordinátának 0-1 közé kellene esnie, mert a limitek a min-max értékek alapján lettek beállítva
        #print('x:' + str(x) + ' x_transzf:' + str(xy[0]))
        aXOut[i]=xy[0]*xunit
        aYOut[i]=xy[1]*yunit
    return (aXOut,aYOut)

def axesunits(unit='point'):
    if unit=='0-1': return (1,1)
    bbox = plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())       # a koordináta tengelyek befoglaló rect-je
    if unit=='inch': return (bbox.width,bbox.height)
    elif unit=='point': return (bbox.width*72,bbox.height*72)


# PLOT
def pltinit(suptitle=None,tight=None,width=None,height=None,left=None,right=None,top=None,bottom=None,nrows=None,ncols=None,
            hspace=None,wspace=None,sharex=True,sharey=True,**params):
    '''
    Diagram rajzolási folyamat inicializálása
    Ha nincs megadva valamelyik adat, akkor marad az alapértelmezett
      
     suptitle:  16-os betűmérettel megjelenő felirat felül  (extra méretezés esetén maradjon üres, a plt.suptitle függvénnyel önálló megadás)
           - ha csak egy subplot van, akkor a title, xlabel és ylabel  is megadható
     width, height:  ablakszélesség a képernyőmérethez képest,  (0-1) közötti szám
     left,right,top,bottom:  rajzolási terület határai az ablakon belül,  (0-1) közötti szám, minden esetben a bal-alsó saroktól számítva)  
           - a felhasználói felület slidereivel kereshető meg az optimális érték
     nrows,ncols:  subplot-ok elrendezése  (ha csak egy subplot, akkor nem kell megadni)
     hspace,wspace:  height-space és width-space a subplot-ok között (0-1, az átlagos subplot mérethez viszonyítva)
     sharex,sharey:  subplot tengelyek összhangolása (a függőleges tick-label-ek csak a bal szélen jelennek meg) 
     tight:  subplot-ok esetén érdemes megadni (nrows,ncols meg van adva).
            True  esetén tightlayout default beállításokkal. A left,right,top,bottom ilyenkor érdektelen.
            Float számérték is megadható:  a subplot-ok közötti vertikális távolságot jellemzi
            1: egy sor maradjon a jelölőknek    2: két sor maradjon a jelölőknek    (default: 1.08)

    params:    (subplot szintű paraméterek;  több subplot esetén általában a pltinitsub függvénnyel javasolt beállítani)
     title:  10-es betűmérettel megjelenő felirat a második sorban
     xlabel,ylabel:  koordináta tengelyek feliratai
    '''

    print('suptitle:' + suptitle)

    subplots= (nrows or ncols)
    if subplots:
        fig, axes = plt.subplots(nrows=(nrows or 1), ncols=(ncols or 1),sharex=sharex,sharey=sharey)
        # plt.sca(axes)

    if width or height: 
        dpi=plt.rcParams['figure.dpi']
        window = plt.get_current_fig_manager().window
        if width:
            screen_x = window.winfo_screenwidth() / dpi
            width=screen_x*width
        else: width=plt.rcParams['figure.figsize'][0]
        if height:
            screen_y = window.winfo_screenheight() / dpi      # inch-ben
            height=screen_y*height
        else: height=plt.rcParams['figure.figsize'][1]
        plt.gcf().set_size_inches(w=width, h=height)

    if suptitle: 
        if tight: 
            x=0.5           # plt.gcf().get_size_inches()[0]/2
            #print('x:' + str(x))
            horizontalalignment='center'
        else:
            x=left or plt.rcParams['figure.subplot.left']
            horizontalalignment='left'
        plt.suptitle(suptitle,x=x, horizontalalignment=horizontalalignment, fontsize=16)



    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom,hspace=hspace,wspace=wspace)

    if params:
        # print('params')
        axindex=None
        if subplots: axindex=0
        title,xlabel,ylabel = dget(params,'title,xlabel,ylabel')
        pltinitsub(axindex,title=title,xlabel=xlabel,ylabel=ylabel,annotinit=False)

    
    config.aAnnotG=[]
    config.aScatterG=[]

    if tight:
        h_pad=None
        if type(tight) in [float,int]: h_pad=tight
        plt.tight_layout(h_pad=h_pad)
    

    # TESZT, számformátumok
    if False:
    #'{:,}'.format(1234)                 # 1,234                 csak vessző vagy underline választható ezres határolónak
    #'{:,}'.format(1234.34)              # 1,234.34

    #'{:,.3f}'.format(12345678.12)       # 12,345,678.120        decdigit,   semmiképpen nincs e-kitevő,   0-k kerülhetnek a végére
    #'{:,.3f}'.format(1)                 # 1.000
    #'{:,.0f}'.format(1)                 # 1

    #'{:,.3g}'.format(1.1233)            # 1.12          összesített digitszám, kerekítéssel    
    #'{:,.3g}'.format(-112.7)             # -113           
    #'{:,.3g}'.format(1123.7)            # 1.12e+03      a kitevő csak akkor jelenik meg, ha a digitszám nem elegendő a teljes megjelenítéshez
    #'{:,.10g}'.format(1123545.7)        # 1,123.7       
    #'{:,.6g}'.format(1123.7)            # 1,123.7       megelégszik kevesebb digittel is (nem ír 0-t a végére)
    #'{:,.5g}'.format(99999)             # 99,999       megelégszik kevesebb digittel is (nem ír 0-t a végére)

    #'{:,.6e}'.format(1123.7)            # 1.123700e+03      decdigit (fix, 0 kerülhet a végére),  e+nn mindig megjelenik
    #'{:,.6e}'.format(1)                 # 1.000000e+00       

    #'{:%}'.format(0.983)                 # 98.300000%       decdigit (default: 6,  0 kerülhet a végére)
    #'{:.0%}'.format(0.987)               # 99%              szorzás százzal (1.0 a bázis;  a % a kapcsos zárójelen belül)

    #'{:.0f}%'.format(98.7)               # 99%              nincs szorzás (a % a kapcsos zárójelen kívül van)
        end

def pltinitsub(axindex=None,title=None,xlabel=None,ylabel=None,annotinit=True,**params):
    # Subplot rajzolási folyamat inicializálása (ha csak egy subplot van, akkor a pltinit is elég)
    # Ha nincs megadva valamelyik adat, akkor marad az alapértelmezett
    #  
    # axindex:  ha nincs subplot, akkor None. Subplot estén a pltinit-ben megadott nrows*ncols subplot közül melyikre vonatkozik (0-bázisú)
    # params:
    #  title:  10-es betűmérettel megjelenő felirat a subplot felett
    #  xlabel,ylabel:  koordináta tengelyek feliratai

    # A subplot aktuálissá tétele  (a plt.gca() erre a subplot-ra mutasson)
    if axindex!=None: plt.sca(plt.gcf().axes[axindex])


    if title!=None: plt.title(title, fontsize=10, x=0, horizontalalignment='left')
    if xlabel!=None: plt.xlabel(xlabel,alpha=0.6)
    if ylabel!=None: plt.ylabel(ylabel,alpha=0.6)
    

    if annotinit:
        config.aAnnotG=[]
        config.aScatterG=[]


def pltshow(annot=True,legend=False,subplot=False,**params):
    '''
     A rajzolás befejezése (záró formázások) és a diagram megjelenítése (subplot=False esetén)
     Kötelező hívni a rajzolási folyamat legvégén (de a formázások nem kötelezőek)
     A subplot-ok végén a pltshowsub függvény hívása javasolt (nem kötelező hívni; megegyezik a subplot=True hívással)
     Ha csak egy subplot van, akkor nem kell külön subplot szintű hívás
    
     params:
      x1,x2,y1,y2:   koordináta határok   (x esetén dátumstring is megadható;   féloldalasan is megadható)
      xtickerstep:   milyen lépésközzel kövessék egymást az x tengely tick-jei  (dátum tengelyre megadható "date" - minimalizált feliratok)
      ticksize:   fontméret (mindkét tengely jelölőire).  Default: 8     0 esetén nem jelenik meg
      tickpad:    milyen messze legyenek a jelölőfeliratok a tengelytől (point-ban).  Default: 2
      xnumformat,ynumformat:  koordinátatengelyek jelölőinek számformátuma
          ','   ='{x:,}        integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline) 
          '3f'  ='{x:,.3f}'    fix decimal digits (0-k lehetnek a végéne, e+00 kitevő semmiképpen, ezres határoló)
          '5g'  ='{x:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
          '4e'  ='{x:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
          '2%'  ='{x:,.2%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
          '2f%' ='{x:,.2f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)
      annot_fontsize:    default 9
      annot_xoffset,annot_yoffset:    csak pozitív érték adható meg  (a FvAnnotateAdd, határozza meg az irányt)
      annot_count:   legfeljebb ennyi label jelenjen meg pozitív és negatív irányban (default:20, külön-külön érvényes a korlát, az alapvonaltól legtávolabbiak jelennek meg)
      annot_ymax,annot_ymin:  a címkék ne kerüljenek megadott y fölé illetve alá (egyik sem kötelező)
      annot_ystack:  azonos y értékhez tartozó címkék lépcsőzetes eltolása

      annotbands:
      annotcircles:
      annottexts:
    '''

    paramsL=params.get('params')    # másodlagos hívás esetén előfordulhat, hogy be van ágyazva
    if paramsL: params=paramsL

    x1,x2,y1,y2,xtickerstep,ytickerstep,ticksize,tickpad,xnumformat,ynumformat, \
                annot_fontsize,annot_xoffset,annot_yoffset,annot_count,annot_ymax,annot_ymin,annot_ystack,annotbands,annotcircles,annottexts \
                  = dget(params,
       'x1,x2,y1,y2,xtickerstep,ytickerstep,ticksize,tickpad,xnumformat,ynumformat,' +
               'annot_fontsize,annot_xoffset,annot_yoffset,annot_count,annot_ymax,annot_ymin,annot_ystack,annotbands,annotcircles,annottexts')

    if x1!=None or x2!=None:     # megadható csak az egyik is
        if type(x1)==str: x1=pd.to_datetime(x1)
        if type(x2)==str: x2=pd.to_datetime(x2)
        x1_,x2_=plt.xlim()
        if x1==None: x1=x1_
        if x2==None: x2=x2_
        plt.xlim(x1,x2)
    if y1!=None or y2!=None:   # megadható csak az egyik is
        y1_,y2_=plt.ylim()
        if y1==None: y1=y1_
        if y2==None: y2=y2_
        plt.ylim(y1,y2)

    if xtickerstep:
        if type(xtickerstep)==str:
            if xtickerstep=='date': 
                plt.gca().xaxis.set_major_formatter(
                        mpl.dates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))
        else:
            plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(xtickerstep))

    if ytickerstep:
        plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytickerstep))

    fontsize0=9
    tickpad0=None
    if subplot: 
        fontsize0=8
        tickpad0=2

    if ticksize!=None:
        plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    elif subplot: plt.gca().tick_params(axis='both', which='major', labelsize=fontsize0)         # default: 8
    
    if tickpad:
        plt.gca().tick_params(axis='both', which='major', pad=tickpad)
    elif subplot: plt.gca().tick_params(axis='both', which='major', pad=tickpad0)         # default: 2
   
    
    if xnumformat: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(FvNumformat(xnumformat,'x')))
    if ynumformat: plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(FvNumformat(ynumformat,'x')))


    # Annotációs sávok rajzolása (annotbands)
        # array of dict(irány,koord,koord2,color,alpha,caption)
        # - irány: 'vert' vagy 'horz'    Keresztirányban kitölti a teljes rajzterületet.
        # - koord,koord2:  forrásadatokhoz igazódó koordináták (dátum esetén "yyyy-MM-dd" formátum is használható)
        # - color,alpha:  a sáv színe és átlátszósága  (default: szürke, 30%)
        # - caption: (opcionális)  a sávon megjelenő felirat (a tájolása megegyezik a sáv tájolásával, felül illetve balra jelenik meg)  
        # - align,fontsize:  a felirat igazítása (left/right) és fontmérete (default 7)
    if annotbands:
        for annotband in annotbands:
            irány,koord,koord2,color,alpha,caption,align,fontsize = dget(annotband,'irány,koord,koord2,color,alpha,caption,align,fontsize')
            if not koord: continue
            if not irány: irány='vert'
            if not color: color='green'
            if not alpha: alpha=0.07
            if not fontsize: fontsize=7
            if not align: align='left'
            if irány=='vert':
                if koord in ['start','begin']: koord=plt.xlim()[0]
                elif koord2=='end': koord2=plt.xlim()[1]
                plt.fill_betweenx(plt.ylim(),koord,koord2,color=color,alpha=alpha)
                if caption:
                    va=kodto(align,'left:bottom//right:top',align)
                    if va=='top': text_y=plt.ylim()[1]
                    else: text_y=plt.ylim()[0] +(plt.ylim()[1]-plt.ylim()[0])/50
                    plt.text(koord,text_y,caption,fontsize=fontsize,color=color,rotation='vertical',va=va,ha='left')
            elif irány=='horz':
                if koord in ['start','begin']: koord2=plt.ylim()[0]
                elif koord2=='end': koord2=plt.ylim()[1]
                plt.fill_between(plt.xlim(),koord,koord2,color=color,alpha=alpha)
                if caption:
                    if va=='right': text_x=plt.xlim()[1]
                    else: text_x=plt.xlim()[0]
                    plt.text(text_x,koord,caption,fontsize=fontsize,color=color,va='center',ha=align)

    # Annotációs körök rajzolása a diagramvonalakon kívül
        # array of dict(x,y,size,color,caption)
        # - x,y:  a pont koordinátái (forrásadatokhoz igazodó)
        # - size: point-ban  (betűmérethez viszoníttható)
        # - color:  a kör színe és átlátszósága  (default: green)
        # - caption: (opcionális)  a körhöz tartozó annotáció
    if annotcircles:
        for annotcircle in annotcircles:
            x,y,size,color,caption = dget(annotcircle,'x,y,size,color,caption')
            FvScatterAdd(x,y,color,size,caption)
        FvScatterPlot(maxsize=100,colorbar=None)

    # Diagramvonalaktól független magyarázó feliratok
        # array of dict(x,y,caption,ha,va,fontsize,alpha,color)
        # x,y:   relatív koordináták a rajzterület bal alsó sarkához viszonyítva (x=-0.2: a rajzterülettől balra, y=1.15: a rajzterület felett)
        # ha: horizontal alignment, 'left' 'center' 'right'
        # va: vertical alignment, 'top' 'center' 'bottom'
        # color:  'green'   '0.3':szürke árnyalat
    if annottexts:
        for annottext in annottexts:
            x,y,caption,ha,va,fontsize,alpha,color = dget(annottext,'x,y,caption,ha,va,fontsize,alpha,color')
            if not x or not y: continue
            if not ha: ha='left'
            if not va: va='top'
            if not fontsize: fontsize=fontsize0-1
            if not alpha: alpha=0.5
            if subplot: tranform=plt.gca().transAxes
            else: transform=plt.gcf().transFigure
            plt.text(x,y,caption,ha=ha,va=va,fontsize=fontsize,alpha=alpha,transform = transform)
            

    # Annotáció rajzolása (diagram-vonalak illetve adat-pontok annotációja)
    if annot: 
        # print('annot')
        if not annot_fontsize: annot_fontsize=fontsize0
        FvAnnotatePlot(xoffset=annot_xoffset,yoffset=annot_yoffset,fontsize=annot_fontsize,annot_count=annot_count,ymax=annot_ymax,ymin=annot_ymin,ystack=annot_ystack)
    if legend: plt.legend(fontsize=fontsize0)


    if not subplot: plt.show()

def pltshowsub(annot=True,legend=False,**params):
    # Subplot rajzolásának befejezése (záró formázások, annotációk és legend megjelenítése).  
    # A rajzolási folyamat legvégén kötelező hívni a pltshow függvényt
    # Ha csak egy subplot van, akkor nem kell külön subplot szintű hívás
    
    # params:
    #  x1,x2,y1,y2:   koordináta határok   (x esetén dátumstring is megadható;   féloldalasan is megadható)
    #  xtickerstep:   milyen lépésközzel kövessék egymást az x tengely tick-jei  (dátum tengelyre nem jó)
    #  ticksize:   fontméret (mindkét tengely jelölőire).  Default: 8
    #  tickpad:    milyen messze legyenek a jelölőfeliratok a tengelytől (point-ban).  Default: 2
    #  xnumformat,ynumformat:  koordinátatengelyek jelölőinek számformátuma
    #      '{:,}        integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline) 
    #      '{:,.3f}'    fix decimal digits (0-k lehetnek a végéne, e+00 kitevő semmiképpen, ezres határoló)
    #      '{:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
    #      '{:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
    #      '{:,.2%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
    #      '{:,.2f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)


    pltshow(subplot=True,annot=annot,legend=legend,params=params)


def pltpercent(axis='y'):
    # "."     tizedes határoló.  Ha vessző is van előtte, akkor a vessző az ezres határoló (space nem adható meg, legfeljebb "_")
    # "g"     general   Az előtte lévő szám a digitek maximális száma (ha szükséges). A szélessége nem fix, felesleges 0-kat nem ír a végére és az elejére
    # hiba:  a diagram manuális átméretezésekor elromlik 
    #plt.gca().set_yticklabels(['{:.2g}%'.format(x*100) for x in plt.gca().get_yticks()])
    if axis=='y': plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0,decimals=0))
    elif axis=='x': plt.gca().xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0,decimals=0)) 

def FvPlotBkgcolorNegativ(color='0.7',alpha=0.1):
    # Van jobb módszer is ...
    ylimL=plt.ylim()
    xlimL=plt.xlim()
    plt.autoscale(False)         # ne módosítsa az xlim ylim határokat
    plt.fill_between(xlimL,0,ylimL[0],color=color,alpha=alpha)

def normalizeSer(ser,maxout=1):
    maxIn=ser.max()
    minIn=ser.min()
    if abs(maxIn)>abs(minIn) and abs(maxIn)>0: ser=ser*(maxout/abs(maxIn))
    elif abs(minIn)>0: ser=ser*(maxout/abs(minIn))
    return ser

def FvNormalize(na):    # -> naNormalized
    # [-0.5,0.5] sávba transzformálja a tömböt
    max=np.nanmax(na)   # nan értékeket ne vegye figyelembe
    min=np.nanmin(na)
    if abs(max)>abs(min): na=na*0.5/abs(max)
    else: na=na*0.5/abs(min)
    return na


