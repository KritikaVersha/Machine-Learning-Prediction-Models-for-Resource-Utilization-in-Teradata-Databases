from sklearn import linear_model
import csv


##
Log_Filename = "/Users/kritikaversha/Downloads/dbqlog-dump_ready for model fitting.csv"
train_set_x=[]
train_set_y=[]
test_set_x=[]
test_set_y=[]
crossval_set_x=[]
crossval_set_y=[]
count=0
with open(Log_Filename, "rb") as outfile:
    reader = csv.reader(outfile)
    for row in reader:
        for i in range(21):
            row[i]=(row[i]).replace(',','').replace('\r\n','').replace('?','0')
        
        count=count+1
        if ((count>1) and (count<401)):
            train_set_x.append(row[0:6])
            train_set_y.append(row[7:])
        else:
            if (count>240) and (count < 391):
                crossval_set_x.append(row[0:6])
                crossval_set_y.append(row[7:])        
            else:
                if (count >= 391):
                    test_set_x.append(row[0:6])
                    test_set_y.append(row[7:])
train_set_x = [map(int, x) for x in train_set_x]
train_set_y = [map(float, x) for x in train_set_y]
test_set_x = [map(float, x) for x in test_set_x]
test_set_y = [map(float, x) for x in test_set_y]
crossval_set_x = [map(float, x) for x in crossval_set_x]
crossval_set_y = [map(float, x) for x in crossval_set_y]

## Initializing Error Estimation variables
TotalIOCount=[]
AMPCPUTime=[]
ParserCPUTime=[]
MaxAMPCPUTime=[]
MaxAmpIO=[]
SpoolUsage=[]
AMPCPUTimeNorm=[]
ParserCPUTimeNorm=[]
MaxAMPCPUTimeNorm=[]
MaxCPUAmpNumberNorm=[]
SeqRespTime=[]
ReqIOKB=[]
ReqPhysIO=[]
ReqPhysIOKB=[]

##Multivariate input
clf = linear_model.LinearRegression()
clf.fit(train_set_x, train_set_y)
for i in range(len(test_set_x)):
    clf.predict (test_set_x[i])
    TotalIOCount.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
    if ((test_set_y[i])[1]) > 0: 
        AMPCPUTime.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
    if ((test_set_y[i])[2]) > 0: 
        ParserCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
    if ((test_set_y[i])[3]) > 0: 
        MaxAMPCPUTime.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
    if ((test_set_y[i])[4]) > 0: 
        MaxAmpIO.append(((test_set_y[i])[4]-(clf.predict(test_set_x[i])[4]))*100/(test_set_y[i])[4])
    if ((test_set_y[i])[5]) > 0: 
        SpoolUsage.append(((test_set_y[i])[5]-(clf.predict(test_set_x[i])[5]))*100/(test_set_y[i])[5])
    if ((test_set_y[i])[6]) > 0: 
        AMPCPUTimeNorm.append(((test_set_y[i])[6]-(clf.predict(test_set_x[i])[6]))*100/(test_set_y[i])[6])
    if ((test_set_y[i])[7]) > 0: 
        ParserCPUTimeNorm.append(((test_set_y[i])[7]-(clf.predict(test_set_x[i])[7]))*100/(test_set_y[i])[7])
    if ((test_set_y[i])[8]) > 0: 
        MaxAMPCPUTimeNorm.append(((test_set_y[i])[8]-(clf.predict(test_set_x[i])[8]))*100/(test_set_y[i])[8])
    if ((test_set_y[i])[9]) > 0: 
        MaxCPUAmpNumberNorm.append(((test_set_y[i])[9]-(clf.predict(test_set_x[i])[9]))*100/(test_set_y[i])[9])
    if ((test_set_y[i])[10]) > 0: 
        SeqRespTime.append(((test_set_y[i])[10]-(clf.predict(test_set_x[i])[10]))*100/(test_set_y[i])[10])
    if ((test_set_y[i])[11]) > 0: 
        ReqIOKB.append(((test_set_y[i])[11]-(clf.predict(test_set_x[i])[11]))*100/(test_set_y[i])[11])
    if ((test_set_y[i])[12]) > 0: 
        ReqPhysIO.append(((test_set_y[i])[12]-(clf.predict(test_set_x[i])[12]))*100/(test_set_y[i])[12])
    if ((test_set_y[i])[13]) > 0: 
        ReqPhysIOKB.append(((test_set_y[i])[13]-(clf.predict(test_set_x[i])[13]))*100/(test_set_y[i])[13])
        
## Lasso
#clf = linear_model.Lasso()
#clf.fit(train_set_x, train_set_y)
#for i in range(len(test_set_x)):
#    clf.predict (test_set_x[i])
#    TotalIOCount.append(((test_set_y[i])[0]-(clf.predict(test_set_x[i])[0]))*100/(test_set_y[i])[0])
#    if ((test_set_y[i])[1]) > 0: 
#        AMPCPUTime.append(((test_set_y[i])[1]-(clf.predict(test_set_x[i])[1]))*100/(test_set_y[i])[1])
#    if ((test_set_y[i])[2]) > 0: 
#        ParserCPUTime.append(((test_set_y[i])[2]-(clf.predict(test_set_x[i])[2]))*100/(test_set_y[i])[2])
#    if ((test_set_y[i])[3]) > 0: 
#        MaxAMPCPUTime.append(((test_set_y[i])[3]-(clf.predict(test_set_x[i])[3]))*100/(test_set_y[i])[3])
#    if ((test_set_y[i])[4]) > 0: 
#        MaxAmpIO.append(((test_set_y[i])[4]-(clf.predict(test_set_x[i])[4]))*100/(test_set_y[i])[4])
#    if ((test_set_y[i])[5]) > 0: 
#        SpoolUsage.append(((test_set_y[i])[5]-(clf.predict(test_set_x[i])[5]))*100/(test_set_y[i])[5])
#    if ((test_set_y[i])[6]) > 0: 
#        AMPCPUTimeNorm.append(((test_set_y[i])[6]-(clf.predict(test_set_x[i])[6]))*100/(test_set_y[i])[6])
#    if ((test_set_y[i])[7]) > 0: 
#        ParserCPUTimeNorm.append(((test_set_y[i])[7]-(clf.predict(test_set_x[i])[7]))*100/(test_set_y[i])[7])
#    if ((test_set_y[i])[8]) > 0: 
#        MaxAMPCPUTimeNorm.append(((test_set_y[i])[8]-(clf.predict(test_set_x[i])[8]))*100/(test_set_y[i])[8])
#    if ((test_set_y[i])[9]) > 0: 
#        MaxCPUAmpNumberNorm.append(((test_set_y[i])[9]-(clf.predict(test_set_x[i])[9]))*100/(test_set_y[i])[9])
#    if ((test_set_y[i])[10]) > 0: 
#        SeqRespTime.append(((test_set_y[i])[10]-(clf.predict(test_set_x[i])[10]))*100/(test_set_y[i])[10])
#    if ((test_set_y[i])[11]) > 0: 
#        ReqIOKB.append(((test_set_y[i])[11]-(clf.predict(test_set_x[i])[11]))*100/(test_set_y[i])[11])
#    if ((test_set_y[i])[12]) > 0: 
#        ReqPhysIO.append(((test_set_y[i])[12]-(clf.predict(test_set_x[i])[12]))*100/(test_set_y[i])[12])
#    if ((test_set_y[i])[13]) > 0: 
#        ReqPhysIOKB.append(((test_set_y[i])[13]-(clf.predict(test_set_x[i])[13]))*100/(test_set_y[i])[13])
##
#print AMPCPUTime
# # Estimating Error Percentage
print "Accuracy percentage of estimating TotalIOCount: "+str(abs(100-abs(reduce(lambda x, y: x + y, TotalIOCount) / len(TotalIOCount))))
print "Accuracy percentage of estimating AMPCPUTime: "+str(abs(100-abs(reduce(lambda x, y: x + y, AMPCPUTime) / len(AMPCPUTime))))
print "Accuracy percentage of estimating ParserCPUTime: "+str(abs(100-abs(reduce(lambda x, y: x + y, ParserCPUTime) / len(ParserCPUTime))))
print "Accuracy percentage of estimating MaxAMPCPUTime: "+str(abs(100-abs(reduce(lambda x, y: x + y, MaxAMPCPUTime) / len(MaxAMPCPUTime))))
print "Accuracy percentage of estimating MaxAmpIO: "+str(abs(100-abs(reduce(lambda x, y: x + y, MaxAmpIO) / len(MaxAmpIO))))
#print ""+str(abs(100-abs(reduce(lambda x, y: x + y, SpoolUsage) / len(SpoolUsage))))
print "Accuracy percentage of estimating AMPCPUTimeNorm: "+str(abs(100-abs(reduce(lambda x, y: x + y, AMPCPUTimeNorm) / len(AMPCPUTimeNorm))))
print "Accuracy percentage of estimating ParserCPUTimeNorm: "+str(abs(100-abs(reduce(lambda x, y: x + y, ParserCPUTimeNorm) / len(ParserCPUTimeNorm))))
print "Accuracy percentage of estimating MaxAMPCPUTimeNorm: "+str(abs(100-abs(reduce(lambda x, y: x + y, MaxAMPCPUTimeNorm) / len(MaxAMPCPUTimeNorm))))
print "Accuracy percentage of estimating MaxCPUAmpNumberNorm: "+str(abs(100-abs(reduce(lambda x, y: x + y, MaxCPUAmpNumberNorm) / len(MaxCPUAmpNumberNorm))))
print "Accuracy percentage of estimating SeqRespTime: "+str(abs(100-abs(reduce(lambda x, y: x + y, SeqRespTime) / len(SeqRespTime))))
print "Accuracy percentage of estimating ReqIOKB: "+str(abs(100-abs(reduce(lambda x, y: x + y, ReqIOKB) / len(ReqIOKB))))
print "Accuracy percentage of estimating ReqPhysIO: "+str(abs(100-abs(reduce(lambda x, y: x + y, ReqPhysIO) / len(ReqPhysIO))))
print "Accuracy percentage of estimating ReqPhysIOKB: "+str(abs(100-abs(reduce(lambda x, y: x + y, ReqPhysIOKB) / len(ReqPhysIOKB))))
       