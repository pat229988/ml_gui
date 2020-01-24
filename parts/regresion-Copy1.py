import tkinter 
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askretrycancel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class reg:
    dset=[]
    ent=[]
    cl=[]
    def winregloop(self):
        def select_file():
            reg.in_file.delete(0,'end')
            f_name=askopenfilename()
            if(f_name[-4:]!='.csv'):
                askretrycancel(title='invalid file',message='not an .csv file \nDo yoy want to retry')
                f_name=''
            print(f_name)
            reg.in_file.insert(0,f_name)
            reg.dset=pd.read_csv(f_name)
            reg.cl=reg.dset.columns.values
            reg.info.insert(0,reg.cl)
            reg.fcl=reg.cl[:-1]
            reg.lb=reg.cl[-1:]
            l=[]
            for i in range(0,len(reg.fcl)):
                l.append(reg.fcl[i])
            reg.sx.insert(0,reg.fcl)
            reg.sy.insert(0,reg.lb)
            def gtdt():
                for i in range(0,len(reg.fcl)):
                    val=int(en[i].get())
                    reg.ent.append(val)
                x=dict([(l[i],reg.ent[i]) for i in range(0,len(reg.fcl))])
                reg.dset=reg.dset.append(x,ignore_index=True)
                reg.dset=reg.dset.fillna(method='ffill')
                reg.X=reg.dset[reg.fcl].values
                reg.y=reg.dset[reg.lb].values
                xtr,xts,ytr,yts=train_test_split(reg.X,reg.y,test_size=0.2)
                reg.rg=LinearRegression()
                reg.rg.fit(xtr,ytr)
                reg.ypd = reg.rg.predict(xts)
                lypd=reg.ypd[-1:]
                lb_prd=Label(win_re,text='predicted value : ',font=(10),bg='#5AFF00')
                lb_prd.place(x=cl,y=r+80)
                et_prd=Entry(win_re,width=60)
                et_prd.place(x=cl+170,y=r+80)
                et_prd.insert(0,lypd)
            cl=450
            r=200
            la=reg.cl[:-1]
            en=reg.cl[:-1]
            for i in range(0,(len(reg.fcl))):
                la[i]=Label(win_re,text=reg.fcl[i],font=(10),bg='#5AFF00')
                la[i].place(x=cl,y=i*30+r)
                en[i]=Entry(win_re,width=60)
                en[i].place(x=cl+170,y=i*30+r)
            btn=Button(win_re,text='SUBMIT',font=(10),command= gtdt)
            btn.place(x=cl,y=i*34+r)
            r=i*34+r
            
        win_re=Tk()
        win_re.geometry('1800x1200')
        win_re.title('all regression at once')
        frm_re = Frame(win_re,bg='#34495E', width=1800, height=1200)
        frm_re.pack()
        # Canvas
        canvas = Canvas(frm_re, width = 1800, height = 1200, bg='#5AFF00', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1800,1200,anchor=CENTER)
        o=reg()
        reg.sel_file=Button(win_re,text='SELECT',font=(10),command=select_file)
        reg.sel_file.place(x=450,y=70)
        reg.in_file=Entry(win_re,width=50)
        reg.in_file.place(x=600,y=70)
        reg.infol=Label(win_re,text='column titles : ',font=(10),bg='#5AFF00')
        reg.infol.place(x=450,y=110)
        reg.info=Entry(win_re,width=50)
        reg.info.place(x=600,y=110)
        reg.selx=Label(win_re,text='select X : ',font=(10),bg='#5AFF00')
        reg.selx.place(x=450,y=140)
        reg.sx=Entry(win_re,width=20)
        reg.sx.place(x=600,y=140)
        reg.sely=Label(win_re,text='select Y : ',font=(10),bg='#5AFF00')
        reg.sely.place(x=450,y=160)
        reg.sy=Entry(win_re,width=20)
        reg.sy.place(x=600,y=160)
        win_re.mainloop() 