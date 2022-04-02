from ezcharts import *
from ezdate import *

from math import sqrt
import numpy as np

def fn_loss(aInputs,outputs,inweights,outweights):
    # súlyfaktorok összege
    inweightsum=sum(inweights)
    loss=0

    # Ciklus az output értékekre (aInputs elemszáma is ehhez igazodik)
    for i,output in enumerate(outputs):
        inputs=aInputs[i]

        # Átlagolás a különböző forrásokból származó mérésekre
        average=sum(inputs*inweights)/inweightsum
        # average=sum([inputs[j]*inweights[j] for j in range(len(inputs))]) / inweightsum
        diff=average-outputs[i]
        loss += diff*diff

    return sqrt(loss/len(outputs))





def fn_WeightTune(aInputs,outputs,inweights,outweights):   # out: inweights - az input források optimalizált súlyfaktorai
    ''' Súlyfaktorok hangolása
    Az input adatsorok milyen súlyozásával érhető el a legnagyobb egyezés az output adatsorral.

    aInputs:  az output adatok különböző forrásból származó közelítő értékei     np.array(nAdat,nForrás)
        Példa:  közvéleménykutató cégek előrejelzései azokra az időpontokra, amikor választás volt
    outputs:  output adatsor     np.array(nAdat)
        Példa:  választásokon elért tényleges eredmény
    inweights:   kezdőértékek forrásonként.  Ha None, akkor mindegyikre 1 a kezdőérték
    outweights:   (null is lehet)  az output adatok fontosságának vagy megbízhatóságának súlyozása
    '''
    
    sizeAdat = len(aInputs)
    if sizeAdat==0: return

    sizeSource = len(aInputs[0])

    if not inweights: inweights=np.array([1.0]*sizeSource)       # ezt kell hangolni

    # induló loss érték   (cél: loss minimalizálása a súlyfaktorok hangolásával)
    loss=fn_loss(aInputs,outputs,inweights,outweights)

    # Hangolási ciklusok
    nLepeskoz=0.01
    while nLepeskoz>=0.0001:
        loss0=loss
        nCiklus=0       # léptetési ciklus az összes súlyfaktorra
        while loss<loss0 or nCiklus==0:   # kezdj új ciklust, ha legalább az egyik súlyfaktor léptetése csökkenést eredményezett
            loss0=loss
            for j in range(sizeSource):
                saveL=inweights[j]
                # súlyfaktor növelés
                inweights[j]=saveL*(1+nLepeskoz)
                lossL=fn_loss(aInputs,outputs,inweights,outweights)
                bOk = (lossL<loss)
                if not bOk:
                    # súlyfaktor csökkentés
                    inweights[j]=saveL*(1-nLepeskoz)
                    lossL=fn_loss(aInputs,outputs,inweights,outweights)
                    bOk = (lossL<loss)
                if bOk: loss=lossL
                else: inweights[j]=saveL
            nCiklus+=1
        nLepeskoz=nLepeskoz/2;

    return inweights


def fn_inweights(tbl2,adatgazdák,pártok,annotcircles_fidesz,annotcircles_ellenzék,firstdate,lastdate):
    # Kiszámítja a közvéleménykutatók súlyfaktorait, 
    # A teljes időszak minden hónap első napjára kiszámítja a közvéleménykutatók átlagát (pártonként),
    #  majd felveszi ezt az adatsort a tbl2-be egy 'average' nevű közvéleménykutató méréseiként

    adatgazdák=adatgazdák.split(',')         # áttérés list-re a felsorolásos string-ről

    # Előrejelzések adatgazdánkánt és pártonként a választási időpontokra
    előrejelzések={}        # key: adatgazda,párt,dátum
    dátumok_all=pd.date_range('2018.04.01','2022.04.03',freq='MS')
    előrejelzések_all={}    # key: adatgazda,párt     value: series
    for párt in pártok:
        if párt=='Fidesz': választások=annotcircles_fidesz
        elif párt=='Ellenzék': választások=annotcircles_ellenzék

        dátumok=[]
        for választás in választások: dátumok.append(választás['x'])
        for adatgazda in adatgazdák:
            labelToSeries=tblinfo(tbl2,'interpolate',cols=párt,groupby='Adatgazda:' + adatgazda,
                            plttype='gauss',resample=100,gausswidth=15,normalize=False,extend_from=firstdate,extend_to=lastdate)
            ser=labelToSeries[párt]
            
            yvalues=servalueA(ser,dátumok_all)    # interpolációval
            for i in range(len(yvalues)): előrejelzések_all[adatgazda + ',' + párt + ',' + str(dátumok_all[i])] = yvalues[i]

            # A választások napján mért érték számításához külön Gauss átlagolás kell, hogy a választás utáni értékek ne okozzanak torzítást
            for dátum in dátumok:
                labelToSeries=tblinfo(tbl2,'interpolate',cols=párt,
                            query='Dátum<="' + dátum + '" and Adatgazda=="' + adatgazda + '"',
                            extend_from=firstdate,extend_to=dátum,
                            plttype='gauss',resample=50,gausswidth=10,normalize=False)
                ser2=labelToSeries[párt]
                y=ser2.iloc[-1]
                # A Publicus esetén kiugró értékek jelentek meg közvetlenül a választás napján (anomáliának tűnik). Indokolt átlagolni a kétféle mozgóátlagot. 
                if adatgazda=='Publicus':
                    y2=servalue(ser,dátum)
                    y=(y+y2)/2   
                előrejelzések[adatgazda + ',' + párt + ',' + dátum] = y


                            
            # yvalues=servalueA(ser,dátumok)    
            # for i in range(len(yvalues)): előrejelzések[adatgazda + ',' + párt + ',' + dátumok[i]] = yvalues[i]

    # Optimalizációs állományok előállítása
    outputs=[]          # tényleges eredmények a három választáson (Fidesz ill. Ellenzék)
    outweights=[]       # a tényleges eredmények fontosságának súlyozása (nincs használva, egyforma súly)
    aInputs=[]          # a közvéleménykutatók által mért értékek a három választás időpontjában (Fidesz illetve Ellenzék)
                        #   első index: outputs-szal összehangolva,  második index: közvéleménykutatók
    for párt in pártok:
        if párt=='Fidesz': választások=annotcircles_fidesz
        elif párt=='Ellenzék': választások=annotcircles_ellenzék

        for választás in választások:
            dátum,eredmény = dget(választás,'x,y')
            outputs.append(eredmény)
            if dátum=='2018.04.18': outweights.append(1)  # KIIKTATVA a 2018-as parlamenti választás dupla súllyal
            else: outweights.append(1)
            inputs=[]
            for adatgazda in adatgazdák:
                inputs.append(előrejelzések[adatgazda + ',' + párt + ',' + dátum])
            aInputs.append(np.array(inputs))

    
    print('2018-as választáskor mért különbség az ellenzék és a Fidesz között')
    diff2018=[]
    for adatgazda in adatgazdák:
        yEllenzék=előrejelzések[adatgazda + ',Ellenzék,2018.04.18']
        yFidesz=előrejelzések[adatgazda + ',Fidesz,2018.04.18']
        diff2018.append([adatgazda,yEllenzék-yFidesz])
        print(adatgazda,'{:.3g}'.format(yEllenzék-yFidesz))

    print('Az idei választáskor mért különbség az ellenzék és a Fidesz között')
    diff2022=[]
    for adatgazda in adatgazdák:
        yEllenzék=előrejelzések_all[adatgazda + ',Ellenzék,2022-04-01 00:00:00']
        yFidesz=előrejelzések_all[adatgazda + ',Fidesz,2022-04-01 00:00:00']
        diff2022.append([adatgazda,yEllenzék-yFidesz])
        print(adatgazda,'{:.3g}'.format(yEllenzék-yFidesz))



    # Optimalizáció:  a közvéleménykutatók olyan súlyozása, amivel az átlag a lehető legközelebb van a három ismert választás eredményeihez
    inweights = fn_WeightTune(aInputs,np.array(outputs),None,np.array(outweights))

    print('\nKözvéleménykutatók súlyfaktorai (minimum kereséssel)')
    for i,adatgazda in enumerate(adatgazdák):
        print('  ' + adatgazda + ': \t\t' + str(inweights[i]))
    



    # Egy másik súlyozás:   súlyfaktor  =  választási eredményektől való eltérés reciproka (külön-külön a Fideszre és az Ellenzékre vonatkozó mérésekre)  
    diffs=[0]*len(adatgazdák)         # átlagos eltérés az eredményketől közvéleménykutatónként
    # ciklus a kimeneti értékekre
    for i,output in enumerate(outputs):
        # ciklus a közvéleménykutatókra
        for j,input in enumerate(aInputs[i]):
            diffL=output - input
            diffs[j]+=diffL*diffL

    inweights2=[1]*len(adatgazdák)
    eltérésátlagok=[0]*len(adatgazdák)
    for j,diff in enumerate(diffs):
        eltérésátlagok[j]=math.sqrt(diff/len(adatgazdák))
        inweights2[j] = 1 / eltérésátlagok[j]      # szórás reciproka


    print('\nKözvéleménykutatók eltérésátlagai (a súlyfaktorok az eltérésátlagok reciprokai)')
    records=list(zip(adatgazdák,eltérésátlagok))
    records.sort(key = lambda x: x[1])     # Rendezés az eltérésátlag szerint csökkenő sorrendben
    for adatgazda,eltérésátlag in records:
        print('  ' + adatgazda + ': \t\t' + str(eltérésátlag))



    # Harmadik módszer: adatgazdánként és pártonként átlagos eltérés számítás a tényadatokhoz képest
    # ciklus a pártokra
    shifts={}
    for adatgazda in adatgazdák:
        for párt in pártok:
            if párt=='Fidesz': választások=annotcircles_fidesz
            elif párt=='Ellenzék': választások=annotcircles_ellenzék
            diffsum=0
            weightsum=0
            for j,választás in enumerate(választások):
                dátum,eredmény = dget(választás,'x,y')
                weight=1
                if j==0: weight=2       # az országgyűlési választás dupla súllyal
                y=előrejelzések[adatgazda + ',' + párt + ',' + dátum]
                diffsum += (y - eredmény) * weight
                weightsum += weight                
            shifts[párt + ',' + adatgazda]=diffsum/weightsum

    



    # Az eredő görbe elállítása a Fidesz-re illetve az Ellenzékre
    # inweights
    aRec=[]
    weightsum=sum(inweights)
    for dátum in dátumok_all:
        rec=[dátum,'Average']       # képzetes adatgazda: "Average"
        for párt in pártok:
            sumL=0
            for i,adatgazda in enumerate(adatgazdák): sumL+=előrejelzések_all[adatgazda + ',' + párt + ',' + str(dátum)] * inweights[i]
            rec.append(sumL/weightsum)
        aRec.append(rec)
    tblAdd=pd.DataFrame.from_records(aRec,columns=['Dátum','Adatgazda'] + pártok, index=['Dátum'])

    # inweights2-vel
    aRec=[]
    weightsum=sum(inweights2)
    for dátum in dátumok_all:
        rec=[dátum,'Average2']       # képzetes adatgazda: "Average2"
        for párt in pártok:
            sumL=0
            for i,adatgazda in enumerate(adatgazdák): sumL+=előrejelzések_all[adatgazda + ',' + párt + ',' + str(dátum)] * inweights2[i]
            rec.append(sumL/weightsum)
        aRec.append(rec)
    tblAdd2=pd.DataFrame.from_records(aRec,columns=['Dátum','Adatgazda'] + pártok, index=['Dátum'])


    # shifts
    aRec=[]
    for dátum in dátumok_all:
        rec=[dátum,'Eltolással']       # képzetes adatgazda: "Eltolással"
        for párt in pártok:
            sumL=0
            for i,adatgazda in enumerate(adatgazdák): sumL+=előrejelzések_all[adatgazda + ',' + párt + ',' + str(dátum)] + shifts[párt + ',' + adatgazda]
            rec.append(sumL/len(adatgazdák))
        aRec.append(rec)
    tblAdd3=pd.DataFrame.from_records(aRec,columns=['Dátum','Adatgazda'] + pártok, index=['Dátum'])



    return pd.concat([tbl2,tblAdd,tblAdd2,tblAdd3])












