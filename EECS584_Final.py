#!/usr/bin/env python
from xml.etree.ElementTree import XML, fromstring, tostring
import re,csv,datetime
import numpy as np
from sklearn.svm import SVR
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import linear_model
from dateutil import parser
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

#######################======================================CSV to XML Conversion==================================================================="   
def extractxml(filename):
    match=[]
    combo=''
    xmlstring=''
    index=0
    dataset_x=[]
    dataset_y=[]
    with open(filename, "rb") as outfile:
        for i in outfile.readlines():
            if 'XMLTextInfo' not in i:
                match.append(i.split('","'))        
    
    for single_row in range(len(match)):
        index=index+1
        if (('<?xml' in (match[single_row])[-1]) and ('</QryPlanXML>' in (match[single_row])[-1])):
            combo=(((match[single_row])[-1])[:-2]).replace('\n','')
            xmlstring = fromstring((combo).encode('utf-16-be'))
        #######################=================================================================================================================================="   
        if (('<?xml' not in (match[single_row])[-1]) and ('</QryPlanXML>' not in (match[single_row-1])[-1])):
                #print ((match[single_row])[0])[2:]
                combo=(((((match[single_row-1])[-1])[:-2])+(match[single_row])[-1])[:-2])
                xmlstring = fromstring((combo).encode('utf-16-be'))
        #######################=================================================================================================================================="   
        else:
            if (('<?xml' not in (match[single_row])[-1]) and ('</QryPlanXML>' not in (match[single_row+1])[-1])):
                #print ((match[single_row])[0])[2:]
                combo=(((((match[single_row+1])[-1])[:-2])+(match[single_row])[-1])[:-2])
                xmlstring = fromstring((combo).encode('utf-16-be'))
        
        parsed_values=parsetree(xmltree=xmlstring)
        parsed_row=xmldata(xmlvalue=parsed_values)
        dataset_x.append(parsed_row)
        response_time=str(((parser.parse((match[single_row])[8]))-(parser.parse((match[single_row])[6])))).split(":")[-1]
        dataset_y.append([float(response_time),int(((match[single_row])[10]).replace(',','')),float(((match[single_row])[11]).replace('','')),int(((match[single_row])[21]).replace(',',''))])
        total_dataset_x=np.array(dataset_x, np.float)
        total_dataset_y=np.array(dataset_y, np.float)
        train_set_x=np.array(dataset_x[0:13], np.float)
        train_set_y=np.array(dataset_y[0:13], np.int)
        test_set_x=np.array(dataset_x[14:], np.float)
        test_set_y=np.array(dataset_y[14:], np.float)
    return [total_dataset_y,total_dataset_x,train_set_x,train_set_y,test_set_x,test_set_y]
        
#######################======================================Parsing the XML================================================================================="   
def parsetree(xmltree):
    d={}
    for i in xmltree.getiterator():
                for key,value in (i.attrib).items():
                    d.setdefault(key,[])
                    if key not in d:
                        d[key]=value
                    else:
                        d[key].append(value)
    return d
#######################======================================Filtering Features================================================================================"   

def xmldata(xmlvalue):
    dataset=[]
    Not_Used=['ReqPhysIO','AMPCPURunDelay','IOKB','AMPCPUTime','MinAmpIO','MaxAMPCPURunDelay','SpoolUsage','ParserCPUTime','SpoolSize','StatementType','MaxAMPOtherWaitTime','PEOtherWaitTime',
              'ReqPhysIOKB','AMPCPUTimeNorm','CPUTime','MinAmpCPUTime','AMPOtherWaitTime','MaxAMPOtherWaitTime','PECPURunDelay',
              'DisCPUTimeNorm','ParserCPUKernelTime','IndexType','MaxAmpCPUTime','PEIOWaitTime','MinAmpCPUTime']
    dataset.append((xmlvalue.get('ReqPhysIO'))[0])
    dataset.append( sum([float(x) for x in xmlvalue.get('AMPCPURunDelay')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('IOKB')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('AMPCPUTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('MinAmpIO')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('MaxAMPCPURunDelay')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('SpoolUsage')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('ParserCPUTime')]))
    #dataset.append( sum([float(x) for x in xmlvalue.get('SpoolSize')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('MaxAMPOtherWaitTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('PEOtherWaitTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('ReqPhysIOKB')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('AMPCPUTimeNorm')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('CPUTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('MinAmpCPUTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('AMPOtherWaitTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('PECPURunDelay')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('DisCPUTimeNorm')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('ParserCPUKernelTime')]))
    dataset.append(len(xmlvalue.get('IndexType')))
    dataset.append( sum([float(x) for x in xmlvalue.get('MaxAMPCPUTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('PEIOWaitTime')]))
    dataset.append( sum([float(x) for x in xmlvalue.get('MinAmpCPUTime')]))
    dataset.append(sum([float(x) for x in xmlvalue.get('MinAmpCPUTime')]))
    stat={}
    count=0
    for i in (xmlvalue.get('StatementType')):
        if i in stat:
            stat[i]+=1
        else:
            stat[i]=1
    dataset.append((stat.values())[0])
    #print xmlvalue.get('Cardinality')
    return dataset   


#######################======================================Ordinary Least Square Regression Model================================================================================"       

def OLS(train_set_x,train_set_y,test_set_x,test_set_y):
    
    
    TotalIOCount=[]
    ResponseTime=[]
    AMPCPUTime=[]
    SpoolUsage=[]
    clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    clf.fit(train_set_x, train_set_y)
    for i in range(len(test_set_x)):
        clf.predict (test_set_x[i])
        if ((test_set_y[i])[0]) > 0: 
            ResponseTime.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
        if ((test_set_y[i])[1]) > 0: 
            TotalIOCount.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
        if ((test_set_y[i])[2]) > 0: 
            AMPCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
        if ((test_set_y[i])[3]) > 0: 
            SpoolUsage.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
    return [ResponseTime,TotalIOCount,AMPCPUTime,SpoolUsage]

#######################======================================Multivariate Regression Model (Normalization, Regularization)================================================================================"       

def MultiVarReg(train_set_x,train_set_y,test_set_x,test_set_y):
    
    
    TotalIOCount=[]
    ResponseTime=[]
    AMPCPUTime=[]
    SpoolUsage=[]
    clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
    clf.fit(train_set_x, train_set_y)
    for i in range(len(test_set_x)):
        clf.predict (test_set_x[i])
        if ((test_set_y[i])[0]) > 0: 
            ResponseTime.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
        if ((test_set_y[i])[1]) > 0: 
            TotalIOCount.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
        if ((test_set_y[i])[2]) > 0: 
            AMPCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
        if ((test_set_y[i])[3]) > 0: 
            SpoolUsage.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
    return [ResponseTime,TotalIOCount,AMPCPUTime,SpoolUsage]

#######################======================================Ridge Regression Model================================================================================"       

def Rid(train_set_x,train_set_y,test_set_x,test_set_y):
    TotalIOCount=[]
    ResponseTime=[]
    AMPCPUTime=[]
    SpoolUsage=[]
    clf = Ridge(alpha=1.0)
    clf.fit(train_set_x, train_set_y)
    #SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.001, kernel='rbf', max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.001, verbose=False)
    for i in range(len(test_set_x)):
        clf.predict (test_set_x[i])
        if ((test_set_y[i])[0]) > 0: 
            ResponseTime.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
        if ((test_set_y[i])[1]) > 0: 
            TotalIOCount.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
        if ((test_set_y[i])[2]) > 0: 
            AMPCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
        if ((test_set_y[i])[3]) > 0: 
            SpoolUsage.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
    return [ResponseTime,TotalIOCount,AMPCPUTime,SpoolUsage]


#######################======================================L1 Based Feature Selection================================================================================"       

def Lasso(train_set_x,train_set_y,test_set_x,test_set_y):
    ## Lasso
    TotalIOCount=[]
    ResponseTime=[]
    AMPCPUTime=[]
    SpoolUsage=[]
    clf = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute='auto', tol=0.0001, warm_start=False)
    clf.fit(train_set_x, train_set_y)
    for i in range(len(test_set_x)):
        clf.predict (test_set_x[i])
        if ((test_set_y[i])[0]) > 0: 
            ResponseTime.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
        if ((test_set_y[i])[1]) > 0: 
            TotalIOCount.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
        if ((test_set_y[i])[2]) > 0: 
            AMPCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
        if ((test_set_y[i])[3]) > 0: 
            SpoolUsage.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
    return [ResponseTime,TotalIOCount,AMPCPUTime,SpoolUsage]

#######################======================================BAR PLOT================================================================================"       
def barplot(ax, dpoints):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 4) numpy array
    '''

    # Aggregation
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]

    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # Space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Set of Bars at each position
    for i,cond in enumerate(conditions):
        indices = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indices]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))

    # x-axis tick labels= Regression Model O/P
    ax.set_xticks(indices)
    ax.set_xticklabels(categories)
    plt.setp(plt.xticks()[1], rotation=90)

    # Axis labels
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Regression Models")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')

if __name__ == "__main__":

    filename='/Users/kritikaversha/downloads/dbqlog-dump.csv'
    [total_data_y,total_data_x,train_set_x,train_set_y,test_set_x,test_set_y]=extractxml(filename)
    
    
    OLS_ResponseTime=(OLS(train_set_x,train_set_y,test_set_x,test_set_y))[0]
    OLS_TotalIOCount=(OLS(train_set_x,train_set_y,test_set_x,test_set_y))[1]
    OLS_AMPCPUTime=(OLS(train_set_x,train_set_y,test_set_x,test_set_y))[2]
    OLS_SpoolUsage=(OLS(train_set_x,train_set_y,test_set_x,test_set_y))[3]
    
    print'=====================================Ordinary Least Square Regression(Without Normalization and Regularization)=============================================================================='
    print "Accuracy percentage of estimating Response Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, OLS_ResponseTime) / len(OLS_ResponseTime))))
    print "Accuracy percentage of estimating Total IO Count: "+str(abs(100-abs(reduce(lambda x, y: x + y, OLS_TotalIOCount) / len(OLS_TotalIOCount))))
    print "Accuracy percentage of estimating AMP CPU Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, OLS_AMPCPUTime) / len(OLS_AMPCPUTime))))
    print "Accuracy percentage of estimating Spool Usage: "+str(abs(100-abs(reduce(lambda x, y: x + y, OLS_SpoolUsage) / len(OLS_SpoolUsage))))
    print '\n'
    
    
    Reg_ResponseTime=(MultiVarReg(train_set_x,train_set_y,test_set_x,test_set_y))[0]
    Reg_TotalIOCount=(MultiVarReg(train_set_x,train_set_y,test_set_x,test_set_y))[1]
    Reg_AMPCPUTime=(MultiVarReg(train_set_x,train_set_y,test_set_x,test_set_y))[2]
    Reg_SpoolUsage=(MultiVarReg(train_set_x,train_set_y,test_set_x,test_set_y))[3]
    print'=====================================Multivariate Regression(Normalization, Regularization)====================================================================='
    print "Accuracy percentage of estimating Response Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Reg_ResponseTime) / len(Reg_ResponseTime))))
    print "Accuracy percentage of estimating Total IO Count: "+str(abs(100-abs(reduce(lambda x, y: x + y, Reg_TotalIOCount) / len(Reg_TotalIOCount))))
    print "Accuracy percentage of estimating AMP CPU Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Reg_AMPCPUTime) / len(Reg_AMPCPUTime))))
    print "Accuracy percentage of estimating Spool Usage: "+str(abs(100-abs(reduce(lambda x, y: x + y, Reg_SpoolUsage) / len(Reg_SpoolUsage))))
    
    Rid_ResponseTime=(Rid(train_set_x,train_set_y,test_set_x,test_set_y))[0]
    Rid_TotalIOCount=(Rid(train_set_x,train_set_y,test_set_x,test_set_y))[1]
    Rid_AMPCPUTime=(Rid(train_set_x,train_set_y,test_set_x,test_set_y))[2]
    Rid_SpoolUsage=(Rid(train_set_x,train_set_y,test_set_x,test_set_y))[3]
    
    
    print '\n'
    print'=====================================Ridge Regression=============================================================================='
    print "Accuracy percentage of estimating Response Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Rid_ResponseTime) / len(Rid_ResponseTime))))
    print "Accuracy percentage of estimating Total IO Count: "+str(abs(100-abs(reduce(lambda x, y: x + y, Rid_TotalIOCount) / len(Rid_TotalIOCount))))
    print "Accuracy percentage of estimating AMP CPU Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Rid_AMPCPUTime) / len(Rid_AMPCPUTime))))
    print "Accuracy percentage of estimating Spool Usage: "+str(abs(100-abs(reduce(lambda x, y: x + y, Rid_SpoolUsage) / len(Rid_SpoolUsage))))
    
    print '\n'
    Lasso_ResponseTime=(Lasso(train_set_x,train_set_y,test_set_x,test_set_y))[0]
    Lasso_TotalIOCount=(Lasso(train_set_x,train_set_y,test_set_x,test_set_y))[1]
    Lasso_AMPCPUTime=(Lasso(train_set_x,train_set_y,test_set_x,test_set_y))[2]
    Lasso_SpoolUsage=(Lasso(train_set_x,train_set_y,test_set_x,test_set_y))[3]
    
    
    print '\n'
    print'=====================================LASSO Regression=============================================================================='
    print "Accuracy percentage of estimating Response Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Lasso_ResponseTime) / len(Lasso_ResponseTime))))
    print "Accuracy percentage of estimating Total IO Count: "+str(abs(100-abs(reduce(lambda x, y: x + y, Lasso_TotalIOCount) / len(Lasso_TotalIOCount))))
    print "Accuracy percentage of estimating AMP CPU Time: "+str(abs(100-abs(reduce(lambda x, y: x + y, Lasso_AMPCPUTime) / len(Lasso_AMPCPUTime))))
    print "Accuracy percentage of estimating Spool Usage: "+str(abs(100-abs(reduce(lambda x, y: x + y, Lasso_SpoolUsage) / len(Lasso_SpoolUsage))))
    
    print '\n'
    print'===================================== Comparison Plot=============================================================================='
    RespOLS=(abs(100-abs(reduce(lambda x, y: x + y, OLS_ResponseTime) / len(OLS_ResponseTime))))
    TotIOLS = (abs(100-abs(reduce(lambda x, y: x + y, OLS_TotalIOCount) / len(OLS_TotalIOCount))))  
    CPUOLS= (abs(100-abs(reduce(lambda x, y: x + y, OLS_AMPCPUTime) / len(OLS_AMPCPUTime))))
    SpoolOLS=(abs(100-abs(reduce(lambda x, y: x + y, OLS_SpoolUsage) / len(OLS_SpoolUsage))))
    RespRid=(abs(100-abs(reduce(lambda x, y: x + y, Rid_ResponseTime) / len(Rid_ResponseTime))))
    TotIRid = (abs(100-abs(reduce(lambda x, y: x + y, Rid_TotalIOCount) / len(Rid_TotalIOCount)))) 
    CPURid=(abs(100-abs(reduce(lambda x, y: x + y, Rid_AMPCPUTime) / len(Rid_AMPCPUTime))))
    SpoolRid=(abs(100-abs(reduce(lambda x, y: x + y, Rid_SpoolUsage) / len(Rid_SpoolUsage))))
    RespMulti=(abs(100-abs(reduce(lambda x, y: x + y, Reg_ResponseTime) / len(Reg_ResponseTime))))
    TotIMulti = (abs(100-abs(reduce(lambda x, y: x + y, Reg_TotalIOCount) / len(Reg_TotalIOCount))))  
    CPUMulti=(abs(100-abs(reduce(lambda x, y: x + y, Reg_AMPCPUTime) / len(Reg_AMPCPUTime))))
    SpoolMulti=(abs(100-abs(reduce(lambda x, y: x + y, Reg_SpoolUsage) / len(Reg_SpoolUsage))))
    RespLasso=(abs(100-abs(reduce(lambda x, y: x + y, Lasso_ResponseTime) / len(Lasso_ResponseTime))))
    TotILasso = (abs(100-abs(reduce(lambda x, y: x + y, Lasso_TotalIOCount) / len(Lasso_TotalIOCount))))  
    CPULasso=(abs(100-abs(reduce(lambda x, y: x + y, Lasso_AMPCPUTime) / len(Lasso_AMPCPUTime))))
    SpoolLasso=(abs(100-abs(reduce(lambda x, y: x + y, Lasso_SpoolUsage) / len(Lasso_SpoolUsage))))
    #['ResponseTime', 'OLS', RespOLS],['TotalIOCount', 'OLS', TotIOLS],['Spool Usage', 'OLS', SpoolOLS],['AMP CPU Usage', 'OLS', CPUOLS],
    dpoints = np.array([
           ['ResponseTime', 'Multivar', RespMulti],
           ['ResponseTime', 'Ridge', RespRid],
           ['ResponseTime', 'Lasso', RespLasso],
           ['TotalIOCount', 'Multivar', TotIMulti],
           ['TotalIOCount', 'Ridge', TotIRid],
           ['TotalIOCount', 'Lasso', TotILasso], 
           ['Spool Usage', 'Multivar', SpoolMulti],
           ['Spool Usage', 'Ridge', SpoolRid],
           ['Spool Usage', 'Lasso', SpoolLasso],
           ['AMP CPU Usage', 'Multivar', CPUMulti],
           ['AMP CPU Usage', 'Ridge', CPURid],
           ['AMP CPU Usage', 'Lasso', CPULasso]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    barplot(ax, dpoints)
    plt.show()
    
    
    
    
    