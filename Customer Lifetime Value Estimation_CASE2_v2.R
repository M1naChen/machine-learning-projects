# Case 2: The customer is unlikely to switch to
# another carrier in months 2-3 (the probability of switching is only 1%), but the probability increases to 6% for the following 12 months, and to 30% after that.

#churn rate=0.01 when t<=3
#churn rate=0.06 when t<=15
#churn rate=0.30 when t>15

##Scenario 1 
##Lifetime Value of a Customer With a Contract: 
contract_period<-24#if a customer churns before the period ends, he will be charged break_contract_fee = 50+10*(contract period remained)

churn_rate1<-0.01 #churn rate=0.01 when t<=3
churn_rate2<-0.06 #churn rate=0.06 when t<=15
churn_rate3<-0.30 #churn rate=0.30 when t>15
period1<-3
period2<-15
#period3 >15
retention_rate1<-1-churn_rate1
retention_rate2<-1-churn_rate2
retention_rate3<-1-churn_rate3

allowance_usage<-300#monthly allowed usage within the contract

beyond_usage_cost_tier1<-0.12#if the contract customer exceeds the usage (300), the extra will be charged at 12 cents/min

monthly_service_cost<-30
monthly_service_charge<-50 #
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
break_contract_fee_v<-rep(0,num_reps)
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
    over_usage<-max(total_usage-allowance_usage,0) #check how many minutes the customer overused
    over_usage_charge<-over_usage*beyond_usage_cost_tier1
    monthly_margin<-monthly_service_charge+over_usage_charge-monthly_service_cost#total profit from this customer
    lifetime_margin[t]<-monthly_margin/discount_rate#clv formula, get the discounted lifetime value 
    num_period=num_period+1
    if(t<=period1){#the retention rate differs by the period
      customer_active<-rbinom(1,1,retention_rate1)}#returns 1 if the customer is active;otherwise 0
    else if(t<=period2){
      customer_active<-rbinom(1,1,retention_rate2)}
    else{
      customer_active<-rbinom(1,1,retention_rate3) 
    }
  }
  customer_life[i]<-num_period
  if(customer_life[i]< contract_period){
    break_contract_fee<-50+10*(contract_period-t+1)#charge break contract fee if the customer left before the contract ends
    lifetime_value[i]<-sum(lifetime_margin)-acquisition_cost+break_contract_fee
    break_contract_fee_v[i]<-break_contract_fee
  }else{
    lifetime_value[i]<-sum(lifetime_margin)-acquisition_cost
  }
  # clv[i]<-sum(lifetime_margin)
  mean_usage[i]<-mean(usage)
}


df_c<-matrix(0,nrow=num_reps,ncol=3)
colnames(df_c)<-c("lifetime_value","customer_life","mean_usage")
df_c[,1]<-lifetime_value
df_c[,2]<-customer_life
df_c[,3]<-mean_usage
#View(df_c)
mean(df_c[,"lifetime_value"])
mean(df_c[,"customer_life"])

plot(lifetime_value~customer_life) 
plot(lifetime_value~mean_usage)
hist(customer_life)
hist(lifetime_value)

###Scenario 2 
##Lifetime Value of a Customer WITHOUT a Contract:

churn_rate1<-0.01 #churn rate=0.01 when t<=3
churn_rate2<-0.06 #churn rate=0.06 when t<=15
churn_rate3<-0.30 #churn rate=0.30 when t>15
period1<-3
period2<-15
#period3 >15
retention_rate1<-1-churn_rate1
retention_rate2<-1-churn_rate2
retention_rate3<-1-churn_rate3

usage_tier1<-200
usage_tier2<-300
#usage_tier3 >300 
usage_price_tier1<-0.25# if the usage is <200, 25 cents/min
usage_price_tier2<-0.20# if the usage is <300, 20 cents/min
usage_price_tier3<-0.15# if the usage is >300, 15 cents/min


monthly_service_cost<-30#Virgin mobile's monthly cost 

#monthly_service_charge= usage*tier_price (different for each tier)
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
    if(t<=period1){#the retention rate differs by the period
      customer_active<-rbinom(1,1,retention_rate1)}#returns 1 if the customer is active;otherwise 0
    else if(t<=period2){
      customer_active<-rbinom(1,1,retention_rate2)}
    else{
      customer_active<-rbinom(1,1,retention_rate3) 
    }
  }
  customer_life[i]<-num_period
  clv[i]<-sum(lifetime_margin)-acquisition_cost
  mean_usage[i]<-mean(usage)
}


df<-matrix(0,nrow=num_reps,ncol=3)
colnames(df)<-c("lifetime_value","customer_life","mean_usage")
df[,1]<-clv
df[,2]<-customer_life
df[,3]<-mean_usage
#View(df)
mean(df[,"lifetime_value"])
mean(df[,"customer_life"])

plot(clv~customer_life) 
plot(clv~mean_usage)
hist(customer_life)
hist(clv)