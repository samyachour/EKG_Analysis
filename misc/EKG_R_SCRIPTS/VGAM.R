install.packages("VGAM")
library("VGAM")
train.FEATURES = train.FEATURES[,1:18]#1:26]
test.FEATURES  = test.FEATURES[,1:18]#1:26]
v=vglm(formula = known_train ~.,family=multinomial(refLevel = "N"),data=train.FEATURES) 

summary(v)

coeff2=as.numeric(coefficients(v))

ind.B1=2*(1:(length(coeff2)/2))-1 #a1
ind.B2=2*(1:(length(coeff2)/2))   #o2

mB1    = coeff2[ind.B1]       ##<-----------hard code into python script
mB2    = coeff2[ind.B2]       ##<-----------hard code into python script


#N=1/(exp(t(a1)%*%newdata) )

write