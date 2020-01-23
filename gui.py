import tkinter as tk
from tkinter import *
from tkinter import font  as tkfont
from tkinter import ttk,PhotoImage
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askretrycancel
from tkinter.ttk import Combobox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (StartPage, PageOne, clasif, reg ):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.geometry('1600x900')
        
        canvas = Canvas(self, width = 1600, height = 900, bg='#58FE03', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1600,900,anchor=CENTER)
        
        def donothing():
            print()
        
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff = 0)
        filemenu.add_command(label = "Start Page", command=lambda: controller.show_frame("StartPage"))
        filemenu.add_command(label = "Selection page", command=lambda: controller.show_frame("PageOne"))
        filemenu.add_command(label = "Classification", command=lambda: controller.show_frame("clasif"))
        filemenu.add_command(label = "Regression", command=lambda: controller.show_frame("reg"))
        
        menubar.add_cascade(label = "File", menu = filemenu)
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label = "Undo", command = donothing)

        editmenu.add_separator()

        editmenu.add_command(label = "Cut", command = donothing)
        editmenu.add_command(label = "Copy", command = donothing)
        editmenu.add_command(label = "Paste", command = donothing)
        editmenu.add_command(label = "Delete", command = donothing)
        editmenu.add_command(label = "Select All", command = donothing)

        menubar.add_cascade(label = "Edit", menu = editmenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label = "Help Index", command = donothing)
        helpmenu.add_command(label = "About...", command = donothing)
        menubar.add_cascade(label = "Help", menu = helpmenu)
        
        controller.config(menu = menubar)

        lbl1 = Label(self,font=("Forte","32"), bg='#2FF1D2',text="Welcome To The Login Page !")
        lbl1.place(x=500, y=75)
        
        # Name
        lbl2 = Label(self,font=("Times","12"),bg='#2FF1D2', text="E-mail")
        lbl2.place(x=600, y=215)
        # Name-Textbox:
        txt1 = Entry(self, width=25)
        txt1.place(x=700, y=215)
        
        # # password:
        lbl3 = Label(self,font=("Times","12"),bg='#2FF1D2', text="Password")
        lbl3.place(x=600, y=255)
        # # Password-Textbox:
        txt2 = Entry(self, width=25)
        txt2.place(x=700, y=255)
        
        btn = Button(self,font=("Times", "12", "bold italic"),bg='#2FF1D2', width=6, height=1, text="Submit",
                     command=lambda: controller.show_frame("PageOne"))
        btn.place(x=700, y=295)

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.geometry('1600x900')
        
        canvas = Canvas(self, width = 1600, height = 900, bg='#58FE03', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1600,900,anchor=CENTER)
        
        
        
        lbl1 = Label(self,font=("Sain Serif","32"), bg='#2FF1D2',text="Select Your Choice for ANALYSIS  !")
        lbl1.place(x=520, y=75)
        
        # Classification Button:
        btn = Button(self,font=("Sain Serif", "14", "bold italic"), width=15, height=1, text="Classification",
                     command=lambda: controller.show_frame("clasif"))
        btn.place(x=520, y=155)

        # Regression Button:
        btn = Button(self,font=("Sain Serif", "14", "bold italic"), width=15, height=1, text="Regression",
                     command=lambda: controller.show_frame("reg"))
        btn.place(x=720, y=155)
        
class clasif(tk.Frame):
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    f_name=''
    l=[0,0,0,0,0,0]
    def __init__(self , parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.geometry('1600x900')
        
        canvas = Canvas(self, width = 1600, height = 900, bg='#58FE03', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1600,900,anchor=CENTER)
        
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        f_name=''
        l=[0,0,0,0,0,0]
        
        # Task with Gui
        lbl = tk.Label(self,font=("Sain Serif",26), bg='#2AB9E9', text="CREATING ALL CLASSIFICATION WITH GUI")
        lbl.place(x=280, y=95)
        
        # Selecting Your Data
        lbl1 = tk.Label(self,font=("Sain Serif",18), bg='#2AB9E9', text="Select a CSV file")
        lbl1.place(x=280, y=155)
        
        clasif.in_file=tk.Button(self,text='SELECT',font=(5),width=(15),command=lambda: clasif.in_f(self))
        clasif.in_file.place(x=280,y=197)
        clasif.inf = tk.Entry(self,width=50)
        clasif.inf.place(x=475,y=205)
        clasif.infol=tk.Label(self,text='column titles : ',font=(10),width=(15),bg='#5AFF00')
        clasif.infol.place(x=280,y=235)
        clasif.info=tk.Entry(self,width=90)
        clasif.info.place(x=475,y=238)
        
        clasif.selx=tk.Label(self,text='select X : ',font=(10),width=(15),bg='#5AFF00')
        clasif.selx.place(x=280,y=275)
        clasif.sx=tk.Entry(self,width=20)
        clasif.sx.place(x=475,y=278)
        clasif.sely=tk.Label(self,text='select Y : ',font=(10),width=(15),bg='#5AFF00')
        clasif.sely.place(x=280,y=300)
        clasif.sy=tk.Entry(self,width=20)
        clasif.sy.place(x=475,y=302)
        
        clasif.l1=tk.Label(self,text='Accuracy on training set ',bg='#5AFF00')
        clasif.l1.place(x=550,y=370)
        clasif.l2=tk.Label(self,text='Accuracy on test set ',bg='#5AFF00')
        clasif.l2.place(x=750,y=370)
        clasif.loreg=tk.Button(self,text='Logistic Regression',font=(10),width=(25),
                               command=lambda: clasif.log_reg(self))
        clasif.loreg.place(x=250,y=405)
        clasif.lr_tr=tk.Entry(self, width=20)
        clasif.lr_tr.place(x=550,y=405)
        clasif.lr_ts=tk.Entry(self, width=20)
        clasif.lr_ts.place(x=750,y=405)
        clasif.dtre=tk.Button(self,text='Decision Tree',font=(10),width=(25),
                              command=lambda: clasif.d_tre(self))
        clasif.dtre.place(x=250,y=450)
        clasif.dtre_tr=tk.Entry(self, width=20)
        clasif.dtre_tr.place(x=550,y=450)
        clasif.dtre_ts=tk.Entry(self, width=20)
        clasif.dtre_ts.place(x=750,y=450)
        clasif.knn_bt=tk.Button(self,text='K-Nearest Neighbors',font=(10),width=(25),
                                command=lambda: clasif.k_nn(self))
        clasif.knn_bt.place(x=250,y=495)
        clasif.knn_tr=tk.Entry(self, width=20)
        clasif.knn_tr.place(x=550,y=495)
        clasif.knn_ts=tk.Entry(self, width=20)
        clasif.knn_ts.place(x=750,y=495)
        clasif.lda=tk.Button(self,text='Linear Discriminant Analysis',font=(10),width=(25),
                             command=lambda: clasif.l_da(self))
        clasif.lda.place(x=250,y=540)
        clasif.lda_tr=tk.Entry(self, width=20)
        clasif.lda_tr.place(x=550,y=540)
        clasif.lda_ts=tk.Entry(self, width=20)
        clasif.lda_ts.place(x=750,y=540)
        clasif.gnb=tk.Button(self,text='Gaussian Naive Bayes',font=(10),width=(25),
                             command=lambda: clasif.g_nb(self))
        clasif.gnb.place(x=250,y=585)
        clasif.gnb_tr=tk.Entry(self, width=20)
        clasif.gnb_tr.place(x=550,y=585)
        clasif.gnb_ts=tk.Entry(self, width=20)
        clasif.gnb_ts.place(x=750,y=585)
        clasif.svm=tk.Button(self,text='Support Vector Machine',font=(10),width=(25),
                             command=lambda: clasif.s_vm(self))
        clasif.svm.place(x=250,y=630)
        clasif.svm_tr=tk.Entry(self, width=20)
        clasif.svm_tr.place(x=550,y=630)
        clasif.svm_ts=tk.Entry(self, width=20)
        clasif.svm_ts.place(x=750,y=630)
        clasif.prdBt=tk.Button(self,text='Pred',font=(10),width=(10),command=lambda: clasif.prd(self))
        clasif.prdBt.place(x=1300,y=360)
        clasif.combo = ttk.Combobox(self, width=(20),font=(10))
        clasif.combo['values']= ("Logistic Regression","Decision Tree","K-Nearest Neighbors",
                                 "Linear Discriminant Analysis","Gaussian Naive Bayes","Support Vector Machine")
        clasif.combo.current(0)
        clasif.combo.place(x=950,y=360)
        clasif.txt_mtx=tk.Text(self,height=8, width=60)
        clasif.txt_mtx.place(x=950,y=405)
        clasif.txt_rpt=tk.Text(self,height=12, width=60)
        clasif.txt_rpt.place(x=950,y=545)
        
    def sel_x(self):
        ch=clasif.combo1.get()
        if((ch not in clasif.sy.get())and (ch not in clasif.sx.get())):
            clasif.sx.insert(0,ch)
            clasif.sx.insert(0,",")
        else:
            askretrycancel(title='invalid selection',message='selected value is alredy present in y or x')
            
    def sel_y(self):
        ch=clasif.combo1.get()
        if((ch not in clasif.sx.get())or(clasif.sy.get().isspace())):
            clasif.sy.insert(0,ch)
            clasif.k=1
        else:   
            askretrycancel(title='invalid selection',message='selected value is alredy present in x ')
        
    def in_f(self):
        f_name=askopenfilename()
        if(f_name[-4:]!='.csv'):
            askretrycancel(title='invalid file',message='not an .csv file \nDo yoy want to retry')
            clasif.inf.delete(0,'end')
            f_name=''
        clasif.inf.insert(0,f_name)
        clasif.dset=pd.read_csv(f_name)
        st=clasif.dset.columns.values
        stri=[]
        for i in range(0,len(st)):
            stri.append(st[i])
        #print(stri)
        #clasif.combo1['values']=stri
        #clasif.combo1.current(0)
        fcl=st[:-1][:-1]
        lb=st[-1:][0]
        clasif.info.insert(0,st)
        
    #def aftersel(self):
        #fcl=clasif.sx.get()
        #fcl=fcl[1:]
        #print(len(fcl))
        #sr=[]
        #sr.append("'")
        #for i in range(0,len(fcl)):
            #if(fcl[i]!=","):
                #sr.append(fcl[i])
            #else:
                #sr.append("' '")
        #sr.append("'")
        #print(sr)
        #sr=''.join(sr)
        #print(sr)
        #st=[]
        
        #fcl=sr
        #lb=clasif.sy.get()
        clasif.sx.insert(0,fcl)
        clasif.sy.insert(0,lb)
        X = clasif.dset[fcl].values
        y = clasif.dset[lb].values
        clasif.X_train, clasif.X_test, clasif.y_train, clasif.y_test = train_test_split(X, y, random_state=0)  
        scaler = MinMaxScaler()
        clasif.X_train = scaler.fit_transform(clasif.X_train)
        clasif.X_test = scaler.transform(clasif.X_test)
    def log_reg(self):
        clasif.lr_tr.delete(0,'end')
        clasif.lr_ts.delete(0,'end')
        clasif.logreg = LogisticRegression()
        clasif.logreg.fit(clasif.X_train, clasif.y_train)
        clasif.lr_tr.insert(0,clasif.logreg.score(clasif.X_train, clasif.y_train))
        clasif.lr_ts.insert(0,clasif.logreg.score(clasif.X_test, clasif.y_test))
        clasif.l[0]=clasif.logreg.score(clasif.X_test, clasif.y_test)
    def d_tre(self):
        clasif.dtre_tr.delete(0,'end')
        clasif.dtre_ts.delete(0,'end')
        clasif.clf = DecisionTreeClassifier().fit(clasif.X_train, clasif.y_train)
        clasif.dtre_tr.insert(0,clasif.clf.score(clasif.X_train, clasif.y_train))
        clasif.dtre_ts.insert(0,clasif.clf.score(clasif.X_test, clasif.y_test))
        clasif.l[1]=clasif.clf.score(clasif.X_test, clasif.y_test)
    def k_nn(self):
        clasif.knn_tr.delete(0,'end')
        clasif.knn_ts.delete(0,'end')
        clasif.knn = KNeighborsClassifier()
        clasif.knn.fit(clasif.X_train, clasif.y_train)
        clasif.knn_tr.insert(0,clasif.knn.score(clasif.X_train, clasif.y_train))
        clasif.knn_ts.insert(0,clasif.knn.score(clasif.X_test, clasif.y_test))
        clasif.l[2]=clasif.knn.score(clasif.X_test, clasif.y_test)
    def l_da(self):
        clasif.lda_tr.delete(0,'end')
        clasif.lda_ts.delete(0,'end')
        clasif.lda = LinearDiscriminantAnalysis()
        clasif.lda.fit(clasif.X_train, clasif.y_train)
        clasif.lda_tr.insert(0,clasif.lda.score(clasif.X_train, clasif.y_train))
        clasif.lda_ts.insert(0,clasif.lda.score(clasif.X_test, clasif.y_test))
        clasif.l[3]=clasif.lda.score(clasif.X_test, clasif.y_test)
    def g_nb(self):
        clasif.gnb_tr.delete(0,'end')
        clasif.gnb_ts.delete(0,'end')
        clasif.gnb = GaussianNB()
        clasif.gnb.fit(clasif.X_train, clasif.y_train)
        clasif.gnb_tr.insert(0,clasif.gnb.score(clasif.X_train, clasif.y_train))
        clasif.gnb_ts.insert(0,clasif.gnb.score(clasif.X_test, clasif.y_test))
        clasif.l[4]=clasif.gnb.score(clasif.X_test, clasif.y_test)
    def s_vm(self):
        clasif.svm_tr.delete(0,'end')
        clasif.svm_ts.delete(0,'end')
        clasif.svm = SVC()
        clasif.svm.fit(clasif.X_train, clasif.y_train)
        clasif.svm_tr.insert(0,clasif.svm.score(clasif.X_train, clasif.y_train))
        clasif.svm_ts.insert(0,clasif.svm.score(clasif.X_test, clasif.y_test))
        clasif.l[5]=clasif.svm.score(clasif.X_test, clasif.y_test)
    def prd(self):
        if(clasif.l[0]==0 or clasif.l[1]==0 or clasif.l[2]==0 or clasif.l[3]==0 or clasif.l[4]==0 or clasif.l[5]==0):
            askretrycancel(title='test not performed',
                           message='one of the test has not yet performed.\nretry after compliting tests...')
        else:
            v=clasif.combo.get()
            if(v=='Logistic Regression'):
                method=clasif.logreg
            elif(v=="Decision Tree"):
                method=clasif.clf
            elif(v=='K-Nearest Neighbors'):
                method=clasif.knn
            elif(v=='Linear Discriminant Analysis'):
                method=clasif.lda
            elif(v=='Gaussian Naive Bayes'):
                method=clasif.gnb
            elif(v=='Support Vector Machine'):
                method=clasif.svm
            clasif.txt_mtx.delete(0.1,'end')
            clasif.txt_rpt.delete(0.1,'end')
            clasif.pred = method.predict(clasif.X_test)
            x=confusion_matrix(clasif.y_test, clasif.pred)
            y=classification_report(clasif.y_test, clasif.pred)
            clasif.txt_mtx.insert(tk.END,x)
            clasif.txt_rpt.insert(tk.END,y)

            
class reg(tk.Frame):
    dset=[]
    ent=[]
    cl=[]
    def __init__(self , parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.geometry('1600x900')
        
        canvas = Canvas(self, width = 1600, height = 900, bg='#58FE03', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1600,900,anchor=CENTER)
        
        reg.sel_file=tk.Button(self,text='SELECT',font=(10),command=lambda: reg.select_file(self))
        reg.sel_file.place(x=450,y=70)
        reg.in_file=tk.Entry(self,width=50)
        reg.in_file.place(x=600,y=70)
        reg.infol=tk.Label(self,text='column titles : ',font=(10),bg='#5AFF00')
        reg.infol.place(x=450,y=110)
        reg.info=tk.Entry(self,width=50)
        reg.info.place(x=600,y=110)
        reg.selx=tk.Label(self,text='select X : ',font=(10),bg='#5AFF00')
        reg.selx.place(x=450,y=140)
        reg.sx=tk.Entry(self,width=20)
        reg.sx.place(x=600,y=140)
        reg.sely=tk.Label(self,text='select Y : ',font=(10),bg='#5AFF00')
        reg.sely.place(x=450,y=160)
        reg.sy=tk.Entry(self,width=20)
        reg.sy.place(x=600,y=160)
        
    def select_file(self):
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
        reg.l=[]
        for i in range(0,len(reg.cl)):
            reg.l.append(reg.cl[i])
        reg.sx.insert(0,reg.fcl)
        reg.sy.insert(0,reg.lb)    
        reg.clp=450
        reg.rp=200
        reg.la=reg.cl
        reg.en=reg.cl
        #print(reg.dset.tail(5))
        for i in range(0,(len(reg.cl))):
            reg.la[i]=tk.Label(self,text=reg.cl[i],font=(10),bg='#5AFF00')
            reg.la[i].place(x=reg.clp,y=i*30+reg.rp)
            reg.en[i]=tk.Entry(self,width=60)
            reg.en[i].place(x=reg.clp+170,y=i*30+reg.rp)
        reg.btn=tk.Button(self,text='SUBMIT',font=(10),command=lambda: reg.gtdt(self))
        reg.btn.place(x=reg.clp,y=i*34+reg.rp)
        reg.rp=i*34+reg.rp
        
            
    def gtdt(self):
        print(reg.dset.shape)
        for i in range(0,len(reg.cl)):
            val=float(reg.en[i].get())
            reg.ent.append(val)
        x=dict([(reg.cl[i],reg.ent[i]) for i in range(0,len(reg.cl))])
        print(x)
        
        reg.dset=reg.dset.append(x,ignore_index=True)
        c01=[]
        c01=reg.dset.columns[reg.dset.isna().all()].tolist()
        print("c01:")
        print(c01)
        reg.dset = reg.dset[reg.dset.columns.difference(c01)]
        print(reg.dset.tail(5))
        print(reg.dset.isnull().sum())
        #reg.cl=reg.dset.columns.values
        #reg.info.insert(0,reg.cl)
        #reg.fcl=reg.cl[:-1]
        #reg.lb=reg.cl[-1:]
        reg.X=reg.dset[reg.fcl].values
        reg.y=reg.dset[reg.lb].values
        xtr,xts,ytr,yts=train_test_split(reg.X,reg.y,test_size=0.2,random_state=0)
        reg.rg=LinearRegression()
        reg.rg.fit(xtr,ytr)
        reg.ypd = reg.rg.predict(xts)
        lypd0=reg.ypd[-1:]
        lypd1=yts[-1:]
        lb_prd0=tk.Label(self,text='predicted value : ',font=(10),bg='#5AFF00')
        lb_prd0.place(x=reg.clp,y=reg.rp+80)
        et_prd0=tk.Entry(self,width=60)
        et_prd0.place(x=reg.clp+170,y=reg.rp+80)
        et_prd0.insert(0,lypd0)
        lb_prd1=tk.Label(self,text='assumed value : ',font=(10),bg='#5AFF00')
        lb_prd1.place(x=reg.clp,y=reg.rp+110)
        et_prd1=tk.Entry(self,width=60)
        et_prd1.place(x=reg.clp+170,y=reg.rp+110)
        et_prd1.insert(0,lypd1)
        
        
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
