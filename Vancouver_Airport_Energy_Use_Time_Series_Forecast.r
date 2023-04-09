library(ggplot2)
library(fpp2)
library(repr)
library(seasonal)
library(forecast)
library(fpp)

mydata<-read.csv(file = 'Energy use at YVR.csv')

head(mydata)#to check a sample of the data

energy<-ts(mydata[,2],start=c(1997,1),frequency = 12)

options(repr.plot.width=12, repr.plot.height=9)
autoplot(energy)+
  ggtitle('Time plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Original data")+
  xlab('Year')+
  ylab('Energy use (in thousands of kWh)')

ggseasonplot(energy)+
  ggtitle('Seasonal plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Original data")+
  xlab('Month')+
  ylab('Energy use (in thousands of kWh)')

ggAcf(energy,lag=36)+
  ggtitle('ACF plot : Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Original data")

fit0 <- stl(energy, t.window=13, s.window="periodic", robust=TRUE)

autoplot(fit0)+
  ggtitle('STL Decomposition: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Original data")

mydata.ts<-ts(mydata,start=c(1997,1),frequency = 12)#convert the whole dataset into time series and observe the patterns
plot(mydata.ts[,c(2,3,4,5,6,7,8)],main="Time plot:energy, mean.temp,total.area,total passenger,domestic.passenger,US.passenger,international.passenger")#skip the first column (month year)

mydata.energy.ts <- ts(mydata$energy, frequency=c(12), start=c(1997))

mydata.energy.ts.train <- window(mydata.energy.ts, end=c(2007,12))
mydata.energy.ts.test <- window(mydata.energy.ts, start=c(2008,1))

mydata.energy.ts.train;mydata.energy.ts.test

options(repr.plot.width=12, repr.plot.height=8)
plot(energy,main='Monthly Energy Use at Vancouver International Airport (YVR) 1997-2010, Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)')
lines(meanf(mydata.energy.ts.train,h=36)$mean,col='blue')
lines(rwf(mydata.energy.ts.train,h=36,drift=TRUE)$mean,col='green')
lines(naive(mydata.energy.ts.train,h=36)$mean,col='red')
lines(snaive(mydata.energy.ts.train,h=36)$mean,col='purple')
legend("topleft",lty=1,col=c('blue','green','red','purple'),legend=c("mean method","drift method","naive method","seasonal naive method"))

accuracy<-rbind(accuracy(meanf(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],
      accuracy(rwf(mydata.energy.ts.train,h=36,drift=TRUE),mydata.energy.ts.test)[2,c(2,3,5,6)],
      accuracy(naive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],
      accuracy(snaive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)])
cbind(c('mean','drift','naive','seasonal naive'),accuracy)

fitMAA<-ets(mydata.energy.ts.train,model='MAA')
summary(fitMAA)

plot(forecast(fitMAA,36),main='Monthly Energy Use at Vancouver International Airport (YVR) 1997-2010, Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)',flwd=3)
lines(mydata.energy.ts.test)
legend("topleft", lty=1, col=c("black","steelblue"), 
       c("Data","Forecasts"),cex=0.9)

#other candidate models
fitAAA<-hw(mydata.energy.ts.train,seasonal='additive',h=36)
fitAAM<-hw(mydata.energy.ts.train,seasonal='multiplicative',h=36)
fitMAM<-ets(mydata.energy.ts.train,model='MAM')

accuracy1<-rbind(accuracy(fitMAA)[1,c(2,3,5,6)],
                accuracy(fitAAA)[1,c(2,3,5,6)],
                accuracy(fitAAM)[1,c(2,3,5,6)],
                accuracy(fitMAM)[1,c(2,3,5,6)])
cbind(c('MAA(final)','AAA','AAM','MAM'),accuracy1)

accuracy2<-rbind(accuracy(forecast(fitMAA,36),mydata.energy.ts.test)[2,c(2,3,5,6)],
                accuracy(fitAAA,mydata.energy.ts.test)[2,c(2,3,5,6)],
                accuracy(fitAAM,mydata.energy.ts.test)[2,c(2,3,5,6)],
                accuracy(forecast(fitMAM,36),mydata.energy.ts.test)[2,c(2,3,5,6)])
cbind(c('MAA(final)','AAA','AAM','MAM'),accuracy2)

options(repr.plot.width=12, repr.plot.height=8)
plot(energy,main='Monthly Energy Use at Vancouver International Airport (YVR) 1997-2010, Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)')
lines(meanf(mydata.energy.ts.train,h=36)$mean,col='blue')
lines(rwf(mydata.energy.ts.train,h=36,drift=TRUE)$mean,col='green')
lines(naive(mydata.energy.ts.train,h=36)$mean,col='red')
lines(snaive(mydata.energy.ts.train,h=36)$mean,col='purple')
lines(forecast(fitMAA,36)$mean,col='orange',lwd=3)
legend("topleft",lty=1,col=c('blue','green','red','purple','orange'),legend=c("mean method","drift method","naive method","seasonal naive method","ETS(M,A,A)"))


mean(fitMAA$residuals)

checkresiduals(fitMAA,lags=24)

fit.arima3<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(0,1,2),include.constant = TRUE)
summary(fit.arima3)# AICc=1531.13

autoplot(mydata.energy.ts.train)+
  ggtitle('Time plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Training data")+
  xlab('Year')+
  ylab('Energy use (in thousands of kWh)')

ggAcf(mydata.energy.ts.train,lag=36)+
  ggtitle('ACF plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle = "Training data")

nsdiffs(mydata.energy.ts.train)

ndiffs(diff(mydata.energy.ts.train,12))

autoplot(diff(diff(mydata.energy.ts.train,12),1))+
  ggtitle('Time plot:Monthly Energy use for the Vancouver International Airport(YVR) 1997-2010',subtitle = "Seasonally and First differenced training data")+
  xlab('Year')+
  ylab('Energy use (in thousands of kWh)')+
  theme(plot.title = element_text(size=12))

ggAcf(diff(diff(mydata.energy.ts.train,12),1),lag=48)+
  ggtitle('ACF plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle =  "Seasonally and First differenced training data")+
  theme(plot.title = element_text(size=12))#ACF

ggPacf(diff(diff(mydata.energy.ts.train,12),1),lag=48)+
  ggtitle('PACF plot: Monthly Energy use for the Vancouver International Airport (YVR) 1997-2010',subtitle =  "Seasonally and First differenced training data")+
  theme(plot.title = element_text(size=12))#PACF

plot(forecast(fit.arima3,h=36),main="Time plot: Monthly Energy use for the Vancouver International Airport(YVR)",sub="Forecasts with prediction intervals from ARIMA(1,1,0)(2,1,0)[12]",xlab="Year", ylab="Energy use (in thousands of kWh)",cex.main=1.5,cex.sub=1.5,flwd=3)
lines(mydata.energy.ts.test)
legend("topleft", lty=1, col=c("black","steelblue"), 
       c("Data","Forecasts"),cex=0.9)

#all the models we tried
fit.arima1<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(2,1,0),include.constant = TRUE)
fit.arima2<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(3,1,0),include.constant = TRUE)
fit.arima3<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(0,1,2),include.constant = TRUE)
fit.arima4<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(0,1,3),include.constant = TRUE)
fit.arima5<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(0,1,2),include.constant = TRUE)
fit.arima6<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(0,1,3),include.constant = TRUE)
fit.arima7<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(2,1,0),include.constant = TRUE)
fit.arima8<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(3,1,0),include.constant = TRUE)

Models<-c("ARIMA(1,1,0)(2,1,0)[12]","ARIMA(1,1,0)(3,1,0)[12]","ARIMA(1,1,0)(0,1,2)[12]","ARIMA(1,1,0)(0,1,3)[12]","ARIMA(0,1,1)(0,1,2)[12]","ARIMA(0,1,1)(0,1,3)[12]","ARIMA(0,1,1)(2,1,0)[12]","ARIMA(0,1,1)(3,1,0)[12]")
cbind(Models,
      rbind(
      round(accuracy(forecast(fit.arima1,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima2,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima4,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima5,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima6,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima7,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima8,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4)))#Training set accuracy

Models<-c("ARIMA(1,1,0)(2,1,0)[12]","ARIMA(1,1,0)(3,1,0)[12]","ARIMA(1,1,0)(0,1,2)[12]","ARIMA(1,1,0)(0,1,3)[12]","ARIMA(0,1,1)(0,1,2)[12]","ARIMA(0,1,1)(0,1,3)[12]","ARIMA(0,1,1)(2,1,0)[12]","ARIMA(0,1,1)(3,1,0)[12]")
cbind(Models,
      rbind(
      round(accuracy(forecast(fit.arima1,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima2,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima4,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima5,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima6,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima7,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima8,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4)))#Test set accuracy

plot(mydata.energy.ts.test,main="Forecasts for Monthly Energy Use at Vancouver International Airport (YVR) 2008-2010",cex.main=1.5,sub="Zoom in test set",cex.sub=1.3,xlab="Year",ylab='Energy use (in thousands of kWh)')#Zoom
#lines(mydata.energy.ts.test)
lines(forecast(fit.arima3,h=36)$mean,col="royalblue")
lines(forecast(fit.arima5,h=36)$mean,col="darkgreen")
legend("topleft", lty=1, col=c("royalblue","darkgreen"), 
       c("ARIMA(1,1,0)(0,1,2)[12]","ARIMA(0,1,1)(0,1,2)[12]"))

plot(energy,main='Monthly Energy Use at Vancouver International Airport (YVR) 1997-2010, Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)')
lines(meanf(mydata.energy.ts.train,h=36)$mean,col='blue')
lines(rwf(mydata.energy.ts.train,h=36,drift=TRUE)$mean,col='green')
lines(naive(mydata.energy.ts.train,h=36)$mean,col='red')
lines(snaive(mydata.energy.ts.train,h=36)$mean,col='purple')
lines(forecast(fit.arima3,h=36)$mean,col='darkorange')
legend("topleft",lty=1,col=c('blue','green','red','purple','darkorange'),legend=c("mean method","drift method","naive method","seasonal naive method","ARIMA(1,1,0)(0,1,2)[12]"))

Methods<-c("mean method","drift method","naive method","seasonal naive method","ARIMA(1,1,0)(0,1,2)[12]")
cbind(Methods,rbind(
                round(accuracy(meanf(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(rwf(mydata.energy.ts.train,h=36,drift=TRUE),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(naive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(snaive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4)))

res<-residuals(fit.arima3)
mean(res)

max(res);min(res)

checkresiduals(fit.arima3,lag=24)

Box.test(residuals(fit.arima3), type="Ljung", lag=24)

plot(mydata.energy.ts.test,main='Monthly Energy Use at Vancouver International Airport (YVR) Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)')
lines(forecast(fit.arima3,h=36)$mean,col='red',lwd=1)
lines(forecast(fitMAA,36)$mean,col='blue',lwd=1)
legend("topleft",lty=1,col=c('blue','red'),legend=c("ETS(M,A,A)","ARIMA(1,1,0)(0,1,2)[12]"))

Methods<-c("mean method","drift method","naive method","seasonal naive method","ETS(M,A,A)","ARIMA(1,1,0)(0,1,2)[12]")
cbind(Methods,rbind(
                round(accuracy(meanf(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(rwf(mydata.energy.ts.train,h=36,drift=TRUE),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(naive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(snaive(mydata.energy.ts.train,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(forecast(fitMAA,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
                round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4)))

final<-ets(energy,alpha=0.8082,beta=0.0001,gamma=0.0001,model='MAA')
plot(forecast(final,h=36),flwd=2,,main='Monthly Energy Use at Vancouver International Airport (YVR) Forecast for 2011-2013',xlab='Year',ylab='Energy use (in thousands of kWh)')
legend("topleft", lty=1, col=c("black","steelblue"), 
       c("Data","Forecasts"),cex=0.9)

plot(mydata.ts[,c(2,3,4,5,6,7,8)],main="Time plot:energy, mean.temp,total.area,total passenger,domestic.passenger,US.passenger,international.passenger")#skip the first column (month year)

options(repr.plot.width=16, repr.plot.height=12)
pairs(mydata.ts[,c(2,3,4,5,6,7,8)])

#other candidate models
fitAAA<-hw(mydata.energy.ts.train,seasonal='additive',h=36)
fitAAM<-hw(mydata.energy.ts.train,seasonal='multiplicative',h=36)
fitMAM<-ets(mydata.energy.ts.train,model='MAM')

plot(mydata.energy.ts.test,main='Monthly Energy Use at Vancouver International Airport (YVR) Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)')
lines(fitAAA$mean,col='blue')
lines(fitAAM$mean,col='green')
lines(forecast(fitMAA,36)$mean,col='red')
lines(forecast(fitMAM,36)$mean,col='purple')
legend("topleft",lty=1,col=c('blue','green','red','purple'),legend=c("ETS(A,A,A)","ETS(A,A,M)","ETS(M,A,A)","ETS(M,A,M)"))

#all the models we tried
fit.arima1<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(2,1,0),include.constant = TRUE)
fit.arima2<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(3,1,0),include.constant = TRUE)
fit.arima3<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(0,1,2),include.constant = TRUE)
fit.arima4<-Arima(mydata.energy.ts.train,order=c(1,1,0), seasonal=c(0,1,3),include.constant = TRUE)
fit.arima5<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(0,1,2),include.constant = TRUE)
fit.arima6<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(0,1,3),include.constant = TRUE)
fit.arima7<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(2,1,0),include.constant = TRUE)
fit.arima8<-Arima(mydata.energy.ts.train,order=c(0,1,1), seasonal=c(3,1,0),include.constant = TRUE)

Models<-c("ARIMA(1,1,0)(2,1,0)[12]","ARIMA(1,1,0)(3,1,0)[12]","ARIMA(1,1,0)(0,1,2)[12]","ARIMA(1,1,0)(0,1,3)[12]","ARIMA(0,1,1)(0,1,2)[12]","ARIMA(0,1,1)(0,1,3)[12]","ARIMA(0,1,1)(2,1,0)[12]","ARIMA(0,1,1)(3,1,0)[12]")
cbind(Models,
      rbind(
      round(accuracy(forecast(fit.arima1,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima2,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima4,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima5,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima6,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima7,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima8,h=36),mydata.energy.ts.test)[1,c(2,3,5,6)],4)))#Training set accuracy

Models<-c("ARIMA(1,1,0)(2,1,0)[12]","ARIMA(1,1,0)(3,1,0)[12]","ARIMA(1,1,0)(0,1,2)[12]","ARIMA(1,1,0)(0,1,3)[12]","ARIMA(0,1,1)(0,1,2)[12]","ARIMA(0,1,1)(0,1,3)[12]","ARIMA(0,1,1)(2,1,0)[12]","ARIMA(0,1,1)(3,1,0)[12]")
cbind(Models,
      rbind(
      round(accuracy(forecast(fit.arima1,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima2,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima3,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima4,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima5,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima6,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima7,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4),
      round(accuracy(forecast(fit.arima8,h=36),mydata.energy.ts.test)[2,c(2,3,5,6)],4)))#Test set accuracy

summary(fit.arima3)

summary(fit.arima4)

summary(fit.arima5)

summary(fit.arima6)

#plot(mydata.energy.ts.train,xlim=c(1997,2011),ylim=c(5000,9000),main="Monthly Energy Use at Vancouver International Airport (YVR) 1997-2010, Forecasts for 2008-2010",cex.main=1.5,cex.sub=1.5)
plot(mydata.energy.ts.test,main='Monthly Energy Use at Vancouver International Airport (YVR) Forecast for 2008-2010',xlab='Year',ylab='Energy use (in thousands of kWh)',ylim=c(6500,9000))

lines(forecast(fit.arima3,h=36)$mean,col="green")
lines(forecast(fit.arima4,h=36)$mean,col="blue")
lines(forecast(fit.arima5,h=36)$mean,col="orange")
lines(forecast(fit.arima6,h=36)$mean,col="red")


legend("topleft", lty=1, col=c("green","blue","orange","red"), 
       c("ARIMA(1,1,0)(0,1,2)[12]","ARIMA(1,1,0)(0,1,3)[12]","ARIMA(0,1,1)(0,1,2)[12]","ARIMA(0,1,1)(0,1,3)[12]"))
