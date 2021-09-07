import os, sys, time
import numpy as np
import pickle

#sys.path.append("../src/")
import utils as ut

# import xfoil_module_1 as xfm
from numpy import genfromtxt
from scipy.interpolate import griddata, interp1d
from scipy import interpolate

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import svm



def my_timer(func):
    #import time
    def wrapper(*args,**kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()-t1
        print(f' Function {func.__name__} ran in {t2} seconds !')
        return result
    return wrapper



class AirfoilDatabase:
    def __init__(self,airfoils_dir,database_file,cl_model_file, cd_model_file, cm_model_file):
        self.airfoils_dir = airfoils_dir
        self.db_file = database_file
        self.m_cl_file = cl_model_file
        self.m_cd_file = cd_model_file
        self.m_cm_file = cm_model_file
        self.db = ut.pload(self.db_file)
        self.m_cl,self.sc_cl = ut.pload(self.m_cl_file)
        self.m_cd,self.sc_cd = ut.pload(self.m_cd_file)
        self.m_cm,self.sc_cm = ut.pload(self.m_cm_file)
        self.clean_database()
        
    def clean_database(self):
        # There are some empty database values that needs to be removed...
        # Otherwise the model still tries to use those
        for i in range(len(self.db)):
            if len(self.db[i]) <= 2 :
                print('Removing : ', self.db[i]['name'])
                self.db.pop(i, None)
            else:
                print('Using : ', self.db[i]['name'])
        
    def set_airfoil_by_name(self,name):
        pass
    
    def set_airfoil_by_number(self,nr):
        self.name = self.db[nr]['name']
        self.G = ut.airfoil_as_feature(self.airfoils_dir+self.name+'.dat')
        
    def set_mission(self, re, cl):
        self._re = re
        self._cl = cl
        
    def get_coeff(self,re,alpha,model,scaler):
        X = np.array([np.hstack([self.G,int(re)/100, alpha])])
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    def get_cl(self,re,alpha):
        return self.get_coeff(re,alpha,model=self.m_cl,scaler=self.sc_cl)

    def get_cd(self,re,alpha):
        return self.get_coeff(re,alpha,model=self.m_cd,scaler=self.sc_cd)

    def get_cm(self,re,alpha):
        return self.get_coeff(re,alpha,model=self.m_cm,scaler=self.sc_cm)
    
    def find_alpha_of_cl(self,re,cl): # this should be written as dynamic programming later !
        a0, a1, k = 0., 1.0, 100.
        for i in range(55): # Right now, it is gradient ?ascent? method 
            if i==0 : cl_0 = self.get_cl(re,a0)
            cl_1 = self.get_cl(re,a1)
            dcl_da = (cl_1-cl_0)/(a1-a0)
            a = a1 + k*(cl-cl_1)*dcl_da
            a0,a1,cl_0 = a1,a, cl_1
            #if abs(a1-a0)<0.0051 : break
            if abs(cl-cl_1)<0.0051 : break
        return a1
    
    def predict(self):
        score = 0.
        a = self.find_alpha_of_cl(self._re, self._cl)
        cl = self.get_cl(self._re,a)
        cd = self.get_cd(self._re,a)
        cm = self.get_cm(self._re,a)

        # Very simple score definition : a bit hack right now...
        # score = abs(cm)/0.03 + cl/cd/50
        # score = cm*cl/cd
        score = cl/cd

        if cm < -0.04 :
            return None, None
        else :
            return self.name , {'cl': cl, 'cd': cd, 'cm':cm, 'clcd':cl/cd, 'score': score}





class Results:
    def __init__(self, limit=10, score='score', reverse=True):
        self._d = {}
        self._limit = limit
        self._score = score
        self._reverse = reverse
    
    def add(self,item,value):
        self._d[item] = value
        self.sort_dict()
    
    def remove_above_limit(self):
        if len(self._d)>self._limit : self._d.popitem()
    
    def set_score(self,score):
        self._score=score
        self.sort_dict()
        
    def sort_dict(self):
        self._d = dict(sorted(self._d.items(), key=lambda x: x[1][self._score], reverse=self._reverse) )
        self.remove_above_limit()
        
    def show(self):
        for i,_item in enumerate(self._d.items()):
            print(i+1, _item[0], _item[1])

        for i,_item in enumerate(self._d.items()):
            print('%s & %.3f  & %.4f & %.3f  & %.2f ' % ( _item[0], _item[1]['cl'], _item[1]['cd'], _item[1]['cm'], _item[1]['clcd']) )

class Mission:
    def __init__(self, re=200.,cl=0.5):
        self.re = re
        self.cl = cl


class Selector():
    def __init__(self,airfoildatabase, results, mission=Mission):
        self._ad =airfoildatabase
        self._res = results
        self._mission = mission
        self.get_mission(self._mission)
    
    def get_mission(self,mission):
        self._re = mission.re
        self._cl = mission.cl

        
    def run(self):
        for nr in self._ad.db.keys():
            self._ad.set_airfoil_by_number(nr)
            self._ad.set_mission(self._re, self._cl)
            item,value = self._ad.predict()
            if item != None :
                self._res.add(item,value)

    def show(self):
        self._res.show()


@my_timer
def main():
    # Location of generated models, databases and airfoil coordinate files :
    airfoils_dir  = '../../Airfoils_selected/'
    database_file = '../database/selected_database.pk'
    cl_model_file = '../models/model_cl.pk'
    cd_model_file = '../models/model_cd.pk'
    cm_model_file = '../models/model_cm.pk'

    # Initialize classes
    ad = AirfoilDatabase(airfoils_dir,database_file,cl_model_file, cd_model_file, cm_model_file)
    r = Results(limit=20, score='score', reverse=True)
    # m = Mission(re=124.,cl=0.427) # Conventional AC
    m = Mission(re=145.,cl=0.27) # Flying-Wing AC
    select = Selector(airfoildatabase=ad, results=r, mission=m)

    # Run and Show the best performing airfoils !
    select.run()
    select.show()




if __name__ == "__main__":
    main()





