# What-if2: consider acquisition cost and number of customer acquired are different for with and without contract
total_marketing_cost = 60000000
# Assume normal distribution of acquired customer
mean_contract_cust = 1000000
sd_contract_cust = 50000
set.seed(123)
total_contract_cust = rnorm(1, mean_contract_cust, sd_no_contract_cust)
contract_acquisition_cost<-total_marketing_cost/total_contract_cust

mean_no_contract_cust = 1500000
sd_no_contract_cust = 75000
set.seed(123)
total_no_contract_cust = rnorm(1,mean_no_contract_cust,sd_no_contract_cust)
no_contract_acquisition_cost<-total_marketing_cost/total_no_contract_cust
#Constant churn rate for contract and non-contract scenarios

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
  acquisition_cost<-contract_acquisition_cost
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

mean(df_c[,"lifetime_value"])
mean(df_c[,"customer_life"])
mean(df_c[,"lifetime_value"])*total_contract_cust#total profit

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
  acquisition_cost<-no_contract_acquisition_cost
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
mean(df_nc[,"lifetime_value"])*total_no_contract_cust#total profit

#plot(clv~customer_life) 
#plot(clv~mean_usage)
hist(customer_life,main="no contract")
hist(df_nc[,"lifetime_value"],main='no contract')
