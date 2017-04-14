#train.FEATURES = train.FEATURES[,c(1,2,4,5,12)]
#test.FEATURES  = test.FEATURES[,c(1,2,4,5,12)]

colnames(test.FEATURES) = colnames(train.FEATURES)
colnames(test.FEATURES)[1] = "known_test"

log_reg.MODEL = glm(known_train~.,family=binomial,data=train.FEATURES)       #log.reg model trained on train.FEATURES
summary(log_reg.MODEL)                                                 #shows pvalues for REG.train model

P.train  = predict(log_reg.MODEL,type="response")                      #predictive probabilities of a drug/indi pair in 'train' being 'known'
P.test   = predict(log_reg.MODEL,type="response",newdata=test.FEATURES)#predictive probabilities of a drug/indi pair in 'test'  being 'known'
r.train  = roc(train.FEATURES$known_train,P.train)                                 #gives result: AUC of 0.9535 for running the model REG.train on FM.train
r.test   = roc(test.FEATURES$known_test,P.test)                                   #gives result: AUC of 0.9435 for running the model REG.train on FM.test

c(as.numeric(r.train$auc),as.numeric(r.test$auc))

othresh  = optimal_thresh(P.test,known_test)  #<-------Hard code into python script

(1,3,5)

