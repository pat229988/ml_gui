from tkinter import *
import tkinter
from tkinter import ttk,PhotoImage
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askretrycancel
from tkinter.ttk import Combobox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class clasif:
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    f_name=''
    l=[0,0,0,0,0,0]
    def __init__(self):
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        f_name=''
        l=[0,0,0,0,0,0]
    def winloop(self):
        win = Tk()
        win.title("All CLASSIFICATIONS")
        win.geometry('1800x1200')
        def donothing():
            filewin = Toplevel(win)
            button = Button(filewin, text="Do nothing button")
            button.pack()
            
        menubar = Menu(win)
        filemenu = Menu(menubar, tearoff = 0)
        filemenu.add_command(label="New", command = donothing)
        filemenu.add_command(label = "Open", command = donothing)
        filemenu.add_command(label = "Save", command = donothing)
        filemenu.add_command(label = "Save as...", command = donothing)
        filemenu.add_command(label = "Close", command = donothing)

        filemenu.add_separator()

        filemenu.add_command(label = "Exit", command = win.quit)
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
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label = "Help Index", command = donothing)
        helpmenu.add_command(label = "About...", command = donothing)
        menubar.add_cascade(label = "Help", menu = helpmenu)

        win.config(menu = menubar)

        frm1 = Frame(win,bg='#34495E', width=1800, height=1200)
        frm1.pack()
        # Canvas
        canvas = Canvas(frm1, width = 1800, height = 1200, bg='#5AFF00', bd=0, highlightthickness=0)       
        canvas.place(x=0,y=0) 
        canvas.create_image(1800,1200,anchor=CENTER)

         # Task with Gui
        lbl = Label(win,font=("Sain Serif",26), bg='#2AB9E9', text="CREATING ALL CLASSIFICATION WITH GUI")
        lbl.place(x=350, y=115)

        # Selecting Your Data
        lbl1 = Label(win,font=("Sain Serif",18), bg='#2AB9E9', text="Select a CSV file")
        lbl1.place(x=350, y=170)
        
        o=clasif()
        clasif.in_file=Button(win,text='SELECT',font=(5),width=(15),command=o.in_f)
        clasif.in_file.place(x=350,y=215)
        clasif.inf = Entry(win,width=50)
        clasif.inf.place(x=515,y=220)
        clasif.infol=Label(win,text='column titles : ',font=(10),width=(15),bg='#5AFF00')
        clasif.infol.place(x=350,y=265)
        clasif.info=Entry(win,width=50)
        clasif.info.place(x=515,y=268)
        clasif.selx=Label(win,text='select X : ',font=(10),width=(15),bg='#5AFF00')
        clasif.selx.place(x=350,y=300)
        clasif.sx=Entry(win,width=20)
        clasif.sx.place(x=515,y=302)
        clasif.sely=Label(win,text='select Y : ',font=(10),width=(15),bg='#5AFF00')
        clasif.sely.place(x=350,y=335)
        clasif.sy=Entry(win,width=20)
        clasif.sy.place(x=515,y=335)
        clasif.l1=Label(win,text='Accuracy on training set ',bg='#5AFF00')
        clasif.l1.place(x=650,y=370)
        clasif.l2=Label(win,text='Accuracy on test set ',bg='#5AFF00')
        clasif.l2.place(x=850,y=370)
        clasif.loreg=Button(win,text='Logistic Regression',font=(10),width=(25),command=o.log_reg)
        clasif.loreg.place(x=350,y=405)
        clasif.lr_tr=Entry(win, width=20)
        clasif.lr_tr.place(x=650,y=405)
        clasif.lr_ts=Entry(win, width=20)
        clasif.lr_ts.place(x=850,y=405)
        clasif.dtre=Button(win,text='Decision Tree',font=(10),width=(25),command=o.d_tre)
        clasif.dtre.place(x=350,y=450)
        clasif.dtre_tr=Entry(win, width=20)
        clasif.dtre_tr.place(x=650,y=450)
        clasif.dtre_ts=Entry(win, width=20)
        clasif.dtre_ts.place(x=850,y=450)
        clasif.knn_bt=Button(win,text='K-Nearest Neighbors',font=(10),width=(25),command=o.k_nn)
        clasif.knn_bt.place(x=350,y=495)
        clasif.knn_tr=Entry(win, width=20)
        clasif.knn_tr.place(x=650,y=495)
        clasif.knn_ts=Entry(win, width=20)
        clasif.knn_ts.place(x=850,y=495)
        clasif.lda=Button(win,text='Linear Discriminant Analysis',font=(10),width=(25),command=o.l_da)
        clasif.lda.place(x=350,y=540)
        clasif.lda_tr=Entry(win, width=20)
        clasif.lda_tr.place(x=650,y=540)
        clasif.lda_ts=Entry(win, width=20)
        clasif.lda_ts.place(x=850,y=540)
        clasif.gnb=Button(win,text='Gaussian Naive Bayes',font=(10),width=(25),command=o.g_nb)
        clasif.gnb.place(x=350,y=585)
        clasif.gnb_tr=Entry(win, width=20)
        clasif.gnb_tr.place(x=650,y=585)
        clasif.gnb_ts=Entry(win, width=20)
        clasif.gnb_ts.place(x=850,y=585)
        clasif.svm=Button(win,text='Support Vector Machine',font=(10),width=(25),command=o.s_vm)
        clasif.svm.place(x=350,y=630)
        clasif.svm_tr=Entry(win, width=20)
        clasif.svm_tr.place(x=650,y=630)
        clasif.svm_ts=Entry(win, width=20)
        clasif.svm_ts.place(x=850,y=630)
        clasif.prd=Button(win,text='Pred',font=(10),width=(10),command=o.prd)
        clasif.prd.place(x=1350,y=360)
        clasif.combo = Combobox(win, width=(30),font=(10))
        clasif.combo['values']= ("Logistic Regression","Decision Tree","K-Nearest Neighbors","Linear Discriminant Analysis","Gaussian Naive Bayes","Support Vector Machine")
        clasif.combo.current(0)
        clasif.combo.place(x=1050,y=360)
        clasif.txt_mtx=tkinter.Text(win,height=8, width=60)
        clasif.txt_mtx.place(x=1050,y=405)
        clasif.txt_rpt=tkinter.Text(win,height=12, width=60)
        clasif.txt_rpt.place(x=1050,y=545)
        win.mainloop()
    def in_f(self):
        f_name=askopenfilename()
        if(f_name[-4:]!='.csv'):
            askretrycancel(title='invalid file',message='not an .csv file \nDo yoy want to retry')
            clasif.inf.delete(0,'end')
            f_name=''
        clasif.inf.insert(0,f_name)
        dset=pd.read_csv(f_name)
        st=dset.columns.values
        fcl=st[:-1][:-1]
        lb=st[-1:][0]
        clasif.info.insert(0,st)
        clasif.sx.insert(0,fcl)
        clasif.sy.insert(0,lb)
        X = dset[fcl].values
        y = dset[lb].values
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
            askretrycancel(title='test not performed',message='one of the test has not yet performed.\nretry after compliting tests...')
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
            clasif.txt_mtx.insert(tkinter.END,x)
            clasif.txt_rpt.insert(tkinter.END,y)