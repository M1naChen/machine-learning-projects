
#Constant churn rate for contract and non-contract scenarios
#####################################################################
# Function: confidence_interval
# What it does: returns the desired confidence interval bounds around a sample average, obtained from a vector of input data
#####################################################################
confidence_interval <- function(vector, interval) {
  # Standard deviation of sample
  vec_sd <- sd(vector)
  # Sample size
  n <- length(vector)
  # Mean of sample
  vec_mean <- mean(vector)
  # Error according to t distribution
  error <- qt((interval + 1)/2, df = n - 1) * vec_sd / sqrt(n)
  # Confidence interval as a vector
  result <- c(vec_mean - error, vec_mean + error)
  return(result)
}
##Scenario 1 
##Lifetime Value of a Customer With a Contract: Constant Churn Rate at 2% for each period
contract_period<-24#if a customer churns before the period ends, he will be charged break_contract_fee = 50+10*(contract period remained)
churn_rate_contract<-0.02#churn rate for contract customers
retention_rate<-1-churn_rate_contract #retention rate for contract customers
allowance_usage<-300 #monthly allowed usage within the contract
beyond_usage_cost_tier1<-0.12#if the contract customer exceeds the usage (300), the extra will be charged at 12 cents/min

monthly_service_cost<-30 #the monthly cost of Virgin mobile for each customer
monthly_service_charge<-50 #the amount Virgin mobile charges for contract customers
annunal_interest_rate<-0.05
monthly_interest_rate<-(1+annunal_interest_rate)^(1/12)-1

maxperiod<-120#months
clv<-c()
life_time<-c()
lifetime_value<-c()
customer_life<-c()
mean_usage<-c()

num_reps<-1000

set.seed(1)

for (i in 1:num_reps){
  lifetime_margin<-rep(0,maxperiod)
  usage<-rep(0,maxperiod)
  acquisition_cost<-runif(1,200,540)#uniform distribution to simulate acquisition cost for each customer
  customer_active=1 #customer confirm active at period 1 
  num_period = 0
  while(customer_active==1 & num_period<maxperiod){
    t=num_period+1
    #returns 1 if the customer is active;otherwise 0
    life_time[t]<-customer_active #record the active status of customer at time t
    discount_rate<-(1+monthly_interest_rate)^t #discount of cash flow at time t 
    total_usage<-rnorm(1,300,20) #generate the monthly usage of the active customer
    usage[t]<-total_usage #record the total usage at time t
    over_usage<-max(total_usage-allowance_usage,0) #check how many minutes the customer overused
    over_usage_charge<-over_usage*beyond_usage_cost_tier1
    monthly_margin<-monthly_service_charge+over_usage_charge-monthly_service_cost#total profit from this customer
    lifetime_margin[t]<-monthly_margin/discount_rate#clv formula, get the discounted lifetime value 
    num_period=num_period+1
    customer_active<-rbinom(1,1,retention_rate)
  }
 # print(lifetime_margin)
  customer_life[i]<-num_period
  if(customer_life[i]< contract_period){
    break_contract_fee<-50+10*(contract_period-t+1)#charge break contract fee if the customer left before the contract ends
  lifetime_value[i]<-sum(lifetime_margin) -acquisition_cost +break_contract_fee
  }else{
  lifetime_value[i]<-sum(lifetime_margin)-acquisition_cost
  }
 # print(lifetime_value[i])
 # clv[i]<-sum(lifetime_margin)
  mean_usage[i]<-mean(usage)
}

df_c<-matrix(0,nrow=num_reps,ncol=3)
colnames(df_c)<-c("lifetime_value","customer_life","mean_usage")
df_c[,1]<-lifetime_value
df_c[,2]<-customer_life
df_c[,3]<-mean_usage
#View(df_c)

avg_clv_c<-mean(df_c[,"lifetime_value"])
ci_clv_c <- confidence_interval(lifetime_value, .95)
plus_minus_clv_c <- round(ci_clv_c[2] - avg_clv_c, 2) #plus_minus is the 95% CI half-width
print(sprintf("The average lifetime value (and 95%% CI half-width) is:$ %.2f +/- %.2f", avg_clv_c, plus_minus_clv_c))

avg_lt_c<-mean(df_c[,"customer_life"])
ci_lt_c <- confidence_interval(customer_life, .95)
plus_minus_lt_c <- round(ci_lt_c[2] - avg_lt_c, 2) #plus_minus is the 95% CI half-width
print(sprintf("The average customer life time (and 95%% CI half-width) is: %.2f month +/- %.2f", avg_lt_c, plus_minus_lt_c))


plot(lifetime_value~customer_life) 
plot(lifetime_value~mean_usage)
hist(customer_life,main='contract')
hist(lifetime_value,main='contract')





###Scenario 2 
##Lifetime Value of a Customer WITHOUT a Contract: Constant Churn Rate at 6%
#non contract is flexible, charges the customers by their practical usage
churn_rate_noncontract<-0.06 #churn rate for non contract customers
retention_rate<-1-churn_rate_noncontract #churn rate for non contract customers
usage_tier1<-200 #different usage tiers for non contract customers
usage_tier2<-300
#usage_tier3 >300 
usage_price_tier1<-0.25# if the usage is <200, 25 cents/min
usage_price_tier2<-0.20# if the usage is <300, 20 cents/min
usage_price_tier3<-0.15# if the usage is >300, 15 cents/min

monthly_service_cost<-30 #Virgin mobile's monthly cost 
#monthly_service_charge= usage*tier_price by tiers

annunal_interest_rate<-0.05
monthly_interest_rate<-(1+annunal_interest_rate)^(1/12)-1

maxperiod<-120#months
clv<-c()
lifetime_margin<-c()
life_time<-c()
lifetime_value<-c()
customer_life<-c()
usage<-c()
mean_usage<-c()

num_reps<-1000

set.seed(123)

for (i in 1:num_reps){
  lifetime_margin<-rep(0,maxperiod)
  usage<-rep(0,maxperiod)
  acquisition_cost<-runif(1,200,540)#uniform distribution to simulate acquisition cost for each customer
  customer_active=1 #customer confirm active at period 1 
  num_period = 0
  
  while(customer_active==1 & num_period<maxperiod){
    t=num_period+1
    #returns 1 if the customer is active;otherwise 0
    life_time[t]<-customer_active #record the active status of customer at time t
    discount_rate<-(1+monthly_interest_rate)^t #discount of cash flow at time t 
    total_usage<-rnorm(1,300,20) #generate the monthly usage of the active customer
    usage[t]<-total_usage #record the total usage at time t
    if(total_usage > usage_tier2){#charge the customer based on the usage range
      usage_charge<-usage_tier1*usage_price_tier1 + (usage_tier2-usage_tier1)*usage_price_tier2 + (total_usage-usage_tier2)*usage_price_tier3
    }
    else if(total_usage > usage_tier1){
      usage_charge<-usage_tier1*usage_price_tier1+(total_usage-usage_tier1)*usage_price_tier2}
    else{
      usage_charge<-total_usage*usage_price_tier1} 
    monthly_margin<-usage_charge-monthly_service_cost 
    lifetime_margin[t]<-monthly_margin/discount_rate#clv formula, get the discounted lifetime value 
    num_period=num_period+1
    customer_active<-rbinom(1,1,retention_rate)
  }
  customer_life[i]<-num_period
  clv[i]<-sum(lifetime_margin)-acquisition_cost
  mean_usage[i]<-mean(usage)
}

df_nc<-matrix(0,nrow=num_reps,ncol=3)
colnames(df_nc)<-c("lifetime_value","customer_life","mean_usage")
df_nc[,1]<-clv
df_nc[,2]<-customer_life
df_nc[,3]<-mean_usage
#View(df_nc)
mean(df_nc[,"lifetime_value"])
mean(df_nc[,"customer_life"])

avg_clv_nc<-mean(df_nc[,"lifetime_value"])
ci_clv_nc <- confidence_interval(df_nc[,"lifetime_value"], .95)
plus_minus_clv_nc <- round(ci_clv_nc[2] - avg_clv_nc, 2) #plus_minus is the 95% CI half-width
print(sprintf("The average lifetime value (and 95%% CI half-width) is:$ %.2f +/- %.2f", avg_clv_nc, plus_minus_clv_nc))

avg_lt_nc<-mean(df_nc[,"customer_life"])
ci_lt_nc <- confidence_interval(customer_life, .95)
plus_minus_lt_nc <- round(ci_lt_nc[2] - avg_lt_nc, 2) #plus_minus is the 95% CI half-width
print(sprintf("The average customer life time (and 95%% CI half-width) is: %.2f month +/- %.2f", avg_lt_nc, plus_minus_lt_nc))

#plot(clv~customer_life) 
#plot(clv~mean_usage)
hist(customer_life,main="no contract")
hist(df_nc[,"lifetime_value"],main='no contract')

#####################
# Whale Plot
#####################

whale.data<-as.data.frame(df_c[order(-df_c[,'lifetime_value'])] )#sort customers decreasing order of CLV
colnames(whale.data)<-"clv.ordered"
whale.data$clvpercent <- 100*(whale.data$clv.ordered/sum(whale.data$clv.ordered))
whale.data$clvcumpercent<-cumsum(whale.data$clvpercent)

whale.data$ncustomer<-seq(1,num_reps)
whale.data$percentcustomers<-100*whale.data$ncustomer/num_reps

par(mfrow=c(1,2))
plot(whale.data$percentcustomers,whale.data$clvcumpercent, type="l",
     ylim=c(0,150),main="Whale Curve for Virgin Mobile(Contract)"
     ,xlab = "Percent of Customers", ylab = "Percent of cumulative profit",cex.main=0.8)
abline(100,0,col=3)
abline(v=whale.data$percentcustomers[which.max(whale.data$clv.ordered<=0)],col=2)

abline(h=whale.data$clvcumpercent[which.max(whale.data$clv.ordered<=0)],col=2)
text(70,110,"max(86,102)",cex=0.8)


plot(whale.data$percentcustomers,whale.data$clv.ordered, type="l",
     main="Rank of CLV",
     xlab = "Percent of Customers", ylab = "CLV")
abline(h=0,col=2)


#for non contract
whale.data2<-as.data.frame(df_nc[order(-df_nc[,'lifetime_value'])] )#sort customers decreasing order of CLV
colnames(whale.data2)<-"clv.ordered"
whale.data2$clvpercent <- 100*(whale.data2$clv.ordered/sum(whale.data2$clv.ordered))
whale.data2$clvcumpercent<-cumsum(whale.data2$clvpercent)

whale.data2$ncustomer<-seq(1,num_reps)
whale.data2$percentcustomers<-100*whale.data2$ncustomer/num_reps

par(mfrow=c(1,2))
plot(whale.data2$percentcustomers,whale.data2$clvcumpercent, type="l",
     ylim=c(0,150),main="Whale Curve for Virgin Mobile (Non contract)"
     ,xlab = "Percent of Customers", ylab = "Percent of cumulative profit",cex.main=0.8)
abline(100,0,col=3)
abline(v=whale.data2$percentcustomers[which.max(whale.data2$clv.ordered<=0)],col=2)
abline(h=whale.data2$clvcumpercent[which.max(whale.data2$clv.ordered<=0)],col=2)
text(37,140,"max(59,130)",cex=0.8)

plot(whale.data2$percentcustomers,whale.data2$clv.ordered, type="l",
     main="Rank of CLV",
     xlab = "Percent of Customers", ylab = "CLV")
abline(h=0,col=2)
