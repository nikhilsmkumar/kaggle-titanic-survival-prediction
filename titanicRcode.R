rm(list=ls())
setwd("C:/Users/NikhilS/Downloads/kaggle")
data= read.csv("train.csv", header = T,stringsAsFactors = F)
test1 = read.csv("test.csv", header = T,stringsAsFactors = F)

# Load packages
library('ggplot2')
library('ggthemes')
library('scales') 
library('dplyr') 
library('randomForest')
library("outliers")
library("caret")

#Exploratory data Analysis
#understand the data type
str(data)
str(test1)
test1$Survived<-0
test1<-test1[,c(1,12,2:11)]
test1$Survived<-as.integer(test1$Survived)
#Convert into proper data types
data$Name = as.character(data$Name)
data$Cabin = as.character(data$Cabin)
data$Survived = as.factor(data$Survived)
data$Ticket = as.character(data$Ticket)
str(data)

test1$Name = as.character(test1$Name)
test1$Cabin = as.character(test1$Cabin)
test1$Survived = as.factor(test1$Survived)
test1$Ticket = as.character(test1$Ticket)
str(test1)
#Look at the block of data
head(data, 10)
tail(data, 10)

head(test1, 10)
tail(test1, 10)

#Let us drive some variables (feature engineering)
# Grab title from passenger names
data$Title = gsub('(.*, )|(\\..*)', '', data$Name)
test1$Title = gsub('(.*, )|(\\..*)', '', test1$Name)

# Show title counts by sex
table(data$Sex, data$Title)
table(test1$Sex, test1$Title)
## create few categories
rare_title = c('Lady', 'the Countess','Capt', 'Col', 'Don', 
               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
rare_title1=c('Rev','Dr','Dona','Col')
# Also reassign mlle, ms, and mme accordingly
data$Title[data$Title == 'Mlle'] = 'Miss' 
data$Title[data$Title == 'Ms'] = 'Miss'
data$Title[data$Title == 'Mme'] = 'Mrs' 
data$Title[data$Title %in% rare_title] = 'Rare Title'

test1$Title[test1$Title == 'Ms'] = 'Miss'
test1$Title[test1$Title %in% rare_title1] = 'Rare Title'
# Show title counts by sex again
table(data$Sex, data$Title)
table(test1$Sex, test1$Title)


# Create a family size variable including the passenger themselves
data$Fsize = data$SibSp + data$Parch + 1
test1$Fsize = test1$SibSp + test1$Parch + 1

#Bin family size
data$FsizeD[data$Fsize == 1] = 'single'
data$FsizeD[data$Fsize > 1 & data$Fsize < 4] = 'small'
data$FsizeD[data$Fsize > 4] = 'large'

test1$FsizeD[test1$Fsize == 1] = 'single'
test1$FsizeD[test1$Fsize > 1 & test1$Fsize < 4] = 'small'
test1$FsizeD[test1$Fsize > 4] = 'large'
#extract first letter from string
data$block = substring(data$Cabin, 1, 1)
test1$block = substring(test1$Cabin, 1, 1)
#replace empty spaces with No cabin
data$block[data$block == ""] = "NoCabin"
test1$block[test1$block == ""] = "NoCabin"
#Missing value analysis
apply(data,2, function(x)sum(is.na(x)))
apply(test1,2, function(x)sum(is.na(x)))
#replace all empty spaces with missing values
data = data.frame(apply(data, 2, function(x) gsub("^$|^ $", NA, x)))
test1 = data.frame(apply(test1, 2, function(x) gsub("^$|^ $", NA, x)))
#test missingness
apply(data,2, function(x)sum(is.na(x)))
apply(test1,2, function(x)sum(is.na(x)))
#store in dataframe
df_missing = data.frame(Variables = colnames(data), 
                        Count = apply(data,2, function(x)sum(is.na(x))))
row.names(df_missing) = NULL
df_missing1 = data.frame(Variables = colnames(test1), 
                         Count = apply(test1,2, function(x)sum(is.na(x))))
row.names(df_missing1) = NULL

#impute missing value with different methods
#Let us start experiment
#Impute missing value with mean/median
data$Age = as.numeric(data$Age)
test1$Age = as.numeric(test1$Age)
# data$Age[is.na(data$Age)] = mean(data$Age, na.rm = TRUE) 
# data$Age[is.na(data$Age)] = median(data$Age, na.rm = TRUE) 

#actual value of data[5,6] = 48, mean = 39.8331, median = 37, KNN = 41.47
#KNN imputation
library(DMwR2)
data = knnImputation(data)

test1 = knnImputation(test1)
#Normalized Data
data$Age = (data$Age - min(data$Age))/(max(data$Age) - min(data$Age))
test1$Age = (test1$Age - min(test1$Age))/(max(test1$Age) - min(test1$Age))
data$Fare = as.numeric(data$Fare)
test1$Fare = as.numeric(test1$Fare)
data$Fare = (data$Fare - min(data$Fare))/(max(data$Fare) - min(data$Fare))
test1$Fare = (test1$Fare - min(test1$Fare))/(max(test1$Fare) - min(test1$Fare))

#Identify the row and remove outlier
outlier_tf = outlier(data$Age, opposite=FALSE)
find_outlier = which(outlier_tf == TRUE)
data = data[-find_outlier, ]

#outlier_tf1 = outlier(test1$Age, opposite=FALSE)
#find_outlier1 = which(outlier_tf1 == TRUE)
#test1 = test1[-find_outlier1, ]
#divide the data into train and test
train = data[sample(nrow(data), 800, replace = F), ]
test = data[!(1:nrow(data)) %in% as.numeric(row.names(train)), ]
train$Ticket<-as.character(train$Ticket)
test1$Ticket<-as.character(test1$Ticket)
test$Ticket<-as.character(test$Ticket)
test1$Parch[test1$Parch==9]<-NA
test1 = knnImputation(test1)
table(test1$Parch)
test1$Parch<-factor(test1$Parch)
train = knnImputation(train)
# Build the model (note: not all possible variables are used)
library(randomForest)
rf_model <- randomForest(factor(Survived) ~   Sex +Age+Fare+Title+Pclass+FsizeD+SibSp+Embarked+Parch+block, data =train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:2)

# Get importance
importance = importance(rf_model)
varImportance = data.frame(Variables = row.names(importance), 
                           Importance = round(importance[ , 'MeanDecreaseGini'], 2))

# Create a rank variable based on importance
rankImportance = varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') + coord_flip() + theme_few()


## Predict using the test set
#prediction <- predict(rf_model, test1)
prediction <- predict(rf_model, test1)
submit <- data.frame(PassengerId = test1$PassengerId, Survived = prediction)
write.csv(submit, file = "th.csv", row.names = FALSE)
prediction <- predict(rf_model, test)
xtab = table(observed = test[,2], predicted = prediction)
confusionMatrix(xtab)


#block and Parch need to solved after finding how to drop levels from a factor variable