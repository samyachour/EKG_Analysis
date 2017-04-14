optimal_thresh = function(po2,o){

#po2=predict(fit2,type="response")
# ROC curve
 # po2=P.test
  #o=known

M=matrix(-1,length(po2),10^4)

p=seq(0.0001,0.9999,10^(-4))
for (i in 1:9999)
  for (j in 1:length(po2)){
    if (po2[j]>p[i]) M[j,i]=1 else M[j,i]=0
  }

Sen=rep(0,9999)
Spe=rep(1,9999)

for (i in 1:9999){
  #Sen[i]=table(M[,i],o)[2,2]/(table(M[,i],o)[1,2]+table(M[,i],o)[2,2])
  #Spe[i]=table(M[,i],o)[1,1]/(table(M[,i],o)[1,1]+table(M[,i],o)[2,1])
  Sen[i] = length(o[o==1 & M[,i]==1])/length(o[o==1])
  Spe[i] = length(o[o==0 & M[,i]==0])/length(o[o==0])
}


# Calculate the optimal threshold
T=(1-Spe)^2+(1-Sen)^2
return(p[which.min(T)])
}