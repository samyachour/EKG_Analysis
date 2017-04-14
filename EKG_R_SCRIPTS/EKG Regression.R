rm(list=ls())
setwd("/Users/AlexB/Desktop/EKG_R/EKG_CSV")
#require(elrm)#exact logistic regression, rare outcome variable
#scale(cars04[1,],cars04.pca$center,cars04.pca$scale)%*%cars04.pca$rotation
#as.numeric((cars04[1,]-cars04.pca$center)/cars04.pca$scale)%*%cars04.pca$rotation
#install.packages("pROC")
#sapply(1:114,function(x) var(train.DATA[,x]) )
require(pROC)
Multi    = F
Validate = T

makebinary = function(x){
  if(x=="~"){x=1}else{x=0} 
  return(x)
}


#A00269 test=c(5.452922, 1.197266 ,2.598963 ,2.352789)
#test = c(5.452922, 1.197266 ,2.598963, 2.352789, 0.1435394)

Aname_OV= read.csv(file="REFERENCE-v2.csv",head=F,sep=",")
res     = read.csv(file="residuals.csv",head=F,sep=",")
waveFM  = (read.csv(file="noise_feat.csv",head=F,sep=",")[-1,-1])

afFM.t  = (read.csv(file="8000_feature_mat.csv",head=T,sep=","))
afFM.t  = afFM.t[,-c(110,111,112,113)]

rownames(afFM.t)   = afFM.t[,1]

ind_L3  = dim(afFM.t)[2]
afDISC  = afFM.t[,c(ind_L3-2,ind_L3-1,ind_L3)]
afFM    = afFM.t[,-c(1:43,ind_L3-2,ind_L3-1,ind_L3)]

class(afDISC[,3])="integer" #makes last column 1/0 instead of T/F

A.name  = as.vector(Aname_OV[,1])
OV      = as.vector(Aname_OV[,2])
names(A.name) = A.name
names(OV)     = A.name

if (Multi==T){
  A.name = A.name[rownames(afFM.t)]
  OV     = OV[rownames(afFM.t)]
}

#known   = as.factor(sapply(OV,function(x) makebinary(x,1)))

noiseFM = cbind(res,waveFM)
###########################################################################
train.DATA  = if (Multi==T){afFM}else{noiseFM}      #<------- set equal to feature matrix here (with no OV var)
test.DATA   = train.DATA

known_train = as.factor(if (Multi==T){OV}else{sapply(OV,function(x) makebinary(x))})
known_test  = known_train

train_ind = 1:(dim(train.DATA)[1])
test_ind  = train_ind


###########################################################################
##only run this chunk for validation purposes
if (Validate == T){
  i=1
  k = 10
  group       = sample(c(rep(1:k,floor(length(A.name)/k)),1:(length(A.name)%%k)))

  train_ind   = which(group!=i)
  test_ind    = which(group==i)

  known_train = known_train[train_ind]
  known_test  = known_test[test_ind]

  train.DATA  = train.DATA[train_ind,]
  test.DATA   = test.DATA[test_ind,]
}
###########################################################################
for (x in 1:(dim(train.DATA)[2]) ) {
  if(class(train.DATA[,x])!="numeric" ){class(train.DATA[,x])="numeric"}
}#85,86,106 107 108, 114,109,111,112
#tt=sapply(1:(dim(train.DATA)[2]),function(x) if(class(train.DATA[,x])!="numeric" ){T}else{F})

#train.DATA = train.DATA[,-c(85,86,106, 107, 108, 114)]
cars04     = train.DATA#[,-c(109,111,112)]
cars04.pca = prcomp(cars04, scale.=TRUE, center=TRUE, retx=TRUE)
a          = (cars04.pca$sdev)^2/sum((cars04.pca$sdev)^2)
#keep on increasing range until between 80-95%
for (ii in 1:length(a)){
  if( sum(a[1:ii]) >=.9 ){
    j = ii
    break
  }
} 
pcaFM    = cars04.pca$x[,1:j] #grabs the first k principal components determined in line above (new feature matrix)

OV_pcaFM = cbind(known_train,as.data.frame(pcaFM))

rownames(OV_pcaFM) = A.name[train_ind]

p2   = as.data.frame(OV_pcaFM)[,1:(j+1)]

train.FEATURES = (p2)#[,1:5] 1 + the number of significant features
test.FEATURES  = cbind(known_test,as.data.frame(t(apply(test.DATA,1, function(x)  as.numeric((x-cars04.pca$center)/cars04.pca$scale)%*%cars04.pca$rotation ))))[,1:(j+1)]

if (Multi==T){
  train.FEATURES = cbind(train.FEATURES,afDISC[train_ind,-3])
  test.FEATURES  = cbind(test.FEATURES,afDISC[test_ind,-3])
}
##################################################################################################
##################################################################################################


#newdata=c(5.452922, 1.197266, 2.598963, 2.352789, 0.1435394)
#Beta_Hat =REG.all$coefficients
#my_model = exp(t(Beta_Hat)%*%newdata)/(1+exp(t(Beta_Hat)%*%newdata))
