#FEAT = list(FEAT.out,FEAT.num,FEAT.int,FEAT.fac)
#names(FEAT) = c("out","num","int","fac")

if (!require(VGAM))
{
  install.packages("./VGAM_1.0-3.tar", repos = NULL)
}

library('VGAM')


h = dim

list = list

class = class


r = function(f,FEAT,FEAT.2=NULL,P=NULL,FEAT.names=NULL,model="vglm"){
  model.tr  = get(paste(c(model,".train"),collapse=""))
  model.pr  = get(paste(c(model,".pred"),collapse=""))
  FEAT = clean.list(FEAT)
  if(!is.null(FEAT.2)){FEAT.2=clean.list(FEAT.2)}
  
  Q = f(FEAT=FEAT,FEAT.2=FEAT.2,FEAT.names=FEAT.names,model.train=model.tr,model.pred=model.pr,P=P)
  return(Q)
}

validate = function(FEAT,FEAT.2,model="vglm",...){
  model.train  = get(paste(c(model,".train"),collapse=""))
  model.pred   = get(paste(c(model,".pred"),collapse=""))
  P            = train(FEAT,model.train)
  FEAT.pred    = pred(FEAT.2,P,model.pred)
  F1           = f1score(FEAT$out[,1],FEAT.pred)
  F1$FEAT.pred = FEAT.pred
  F1$P         = P
  return(F1)
}

train = function(FEAT,model.train = vglm.train,pca=T,...){ #INPUT: list (of data.frames), function (model)
  library("VGAM")
  PCA       = pca.train(FEAT$num)
  FEAT.out  = FEAT$out
  FEAT.jpca = if(pca==T){PCA$FEAT.jpca}else{FEAT$num}
  FEAT.intfac=do.call(cbind, FEAT[-c(1,2)])
  FEAT.mat  = cbind(FEAT.out,FEAT.jpca)
  if(!is.null(FEAT.intfac)){FEAT.mat = cbind(FEAT.mat,FEAT.intfac)}
  
  MODEL     = model.train(FEAT.mat)
  out.names = list(out.names=names(summary(FEAT$out[,1])))
  P         = c(PCA,MODEL,out.names)
  return(P)
}#OUTPUT: LIST (of coefficents and matrices for pred function)

pred = function(FEAT,P,model.pred = vglm.pred,...){ #INPUT: list (of data.frames), function (model)
  FEAT.jpca   = pca.pred(FEAT$num,P)
  FEAT.intfac = do.call(cbind, FEAT[-c(1,2)])
  FEAT.best   = if(!is.null(FEAT.intfac)){cbind(FEAT.jpca,FEAT.intfac)}else{FEAT.jpca}
  FEAT.best   = FEAT.best[,P$feat_ind]
  PRED        = model.pred(FEAT.best,P)
  FEAT.pred   = PRED$FEAT.pred
  return(FEAT.pred)
}#OUTPUT factor

###########################################################################################################
###########################################################################################################
f1score   = function(FEAT.out,FEAT.pred){ #INPUT: vec (fac), vec (fac)
  con_mat = table(FEAT.out,FEAT.pred)
  vec     = 2*diag(con_mat)/(colSums(con_mat)+rowSums(con_mat))
  score   = mean(vec)
  F1      = list(score=score,vec=vec,con_mat=con_mat)
  return(F1)
}#OUTPUT: LIST: number (num), vec (num)

clean.list = function(FEAT){ #INPUT: LIST (data.frames,vectors), functions (as.)
  notnull_ind = which(!sapply(FEAT,function(x) is.null(x)))
  FEAT = FEAT[notnull_ind]
  as.L = c(as.factor,as.numeric,as.integer,as.factor)[notnull_ind]
  FEAT = lapply(1:length(FEAT[notnull_ind]),function(x) clean(FEAT[[x]],FEAT$out,as.L[[x]]) )
  names(FEAT) = c("out","num","int","fac")[notnull_ind]
  return(FEAT)
} #OUTPUT: LIST (of data.frames)

############################################################################# 
clean = function(DF,FEAT.out=1,as.class){ #INPUT: factor, data.frame (or matrix, factor, vector), function (as.)
  if ( (length(FEAT.out)==1) && (is.null(dim(DF)) || dim(DF)[2]==1) ){DF=t(DF)}
  if (class(DF)!="data.frame"){DF=as.data.frame(DF)}
  DF = apply(DF,2,function(x) as.class(x))
  if (class(DF)!="data.frame" && class(DF)!="factor"){DF=as.data.frame(DF)}
  return(DF)
} #OUTPUT: data.frame

############################################################################
pca.train = function(FEAT.num,v_thresh=.9){ #INPUT: data.frame (num), number (num)
  FEAT.pca  = prcomp(FEAT.num, scale.=TRUE, center=TRUE, retx=TRUE)
  variances = (FEAT.pca$sdev)^2/sum((FEAT.pca$sdev)^2)
  for (j in 1:length(variances)){if(sum(variances[1:j]) >= v_thresh){break}}
  #keep on increasing range until between 80-95%
  FEAT.jpca = as.data.frame(FEAT.pca$x[,1:j]) #grabs the first k principal components determined in line above (new feature matrix)
  PCA       = list(FEAT.jpca,FEAT.pca$rotation,FEAT.pca$center,FEAT.pca$scale,j)
  names(PCA)= c("FEAT.jpca","rotation","center","scale","j")
  return(PCA)
}#OUTPUT: LIST: data.frame (num), matrix, vec (num), vec (num), integer 

#############################################################################
vglm.train = function(FEAT.mat){ #INPUT: data.frame (all)
  require("VGAM")
  class.num  = length(summary(FEAT.mat[,1]))-1  #number of classes minus 1
  all_feat   = as.formula(paste(names(FEAT.mat)[1], "~."))
  m1         = vglm(formula = all_feat,family=multinomial,data=FEAT.mat) 
  pvals      = summary(m1)@coef3[,4][-(1:class.num)]
  print(pvals)
  pvals.mat  = matrix(pvals,class.num,dim(FEAT.mat)[2]-1)
  print(pvals.mat)
  feat_ind   = which(apply(pvals.mat,2,function(x) all(x<=.05)))
  if(length(feat_ind)==1){print(1)} else if (length(feat_ind)<1){print(2)}
  best_feat  = as.formula(paste(paste(names(FEAT.mat)[1], "~"), paste(names(FEAT.mat)[-1][feat_ind], collapse ="+")))
  
  m2         = vglm(formula = best_feat,family=multinomial,data=FEAT.mat) 
  coeffs     = matrix(coefficients(m2),class.num,length(feat_ind)+1)
  vglm_L        = list(B.mat=coeffs,feat_ind=feat_ind)
  names(vglm_L) = c("B.mat","feat_ind") 
  return(vglm_L) 
}#OUTPUT: LIST: matrix, vector

#############################################################################
pca.pred = function(FEAT.num,P){ #INPUT: data.frame (num), list (of parameters) 
  if (is.null(dim(FEAT.num)) || dim(FEAT.num)[2]==1 ){FEAT.num=t(FEAT.num)}
  FEAT.jpca = as.data.frame(t(apply(FEAT.num,1,function(x) as.numeric((x-P$center)/P$scale))) %*% P$rotation)[,1:P$j]
  return(FEAT.jpca) 
}#OUTPUT: data.frame (num)

#############################################################################
vglm.pred = function(FEAT.best,P){ #INPUT : data.frame (features), LIST (of parameters)
  if (is.null(dim(FEAT.best)) || dim(FEAT.best)[2]==1 ){FEAT.num=t(FEAT.best)}
  vglm.pred.v1  = function(v,B.mat){
    v1          = c(1,v)
    exp_dot     = function(v1,B){exp(sum(v1*B))}
    denom       = sum(apply( B.mat,1,function(x) exp_dot(v1,x)))
    pred_probs  = c( apply( B.mat,1,function(x) exp_dot(v1,x)/denom ) , 1/denom )
  
    return( pred_probs  )
  }
  FEAT.ppmat = t(apply(FEAT.best,1,function(x) vglm.pred.v1(x,P$B.mat)))
  FEAT.pred  = as.factor(P$out.names[t(apply(FEAT.ppmat,1, function(x) which.max(x) ))])
  vglm.pred_L        = list(FEAT.pred,FEAT.ppmat)
  names(vglm.pred_L) = c("FEAT.pred","FEAT.ppmat")
  return(vglm.pred_L) 
}#OUTPUT: LIST: factor, matrix

###########################################################################