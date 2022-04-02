from datetime import datetime


aAnnotG=[]
aScatterG=[]

linecolorsG={'Hungary':{'color':'orange','linewidth':2, 'alpha':1},
             'World':{'color':'0.7','linewidth':4,'alpha':0.3},
             'European Union':{'color':'blue','linewidth':2,'alpha':1},
             'Regisztrációk':{'color':'orange','linewidth':2, 'alpha':1},
             'Átlag':{'color':'orange','linewidth':2, 'alpha':1}}
annotcolorG={'Hungary':'red',
             'European Union':'blue'}

serplotlastG=None        # a FvPlot által utoljára rajzolt görbe 
tblinfoplotsG=None       # a tblinfo 'plot' által kirajzolt series objektumok

d_honapok={'január':1,'február':2,'március':3,'április':4,'május':5,'június':6}


