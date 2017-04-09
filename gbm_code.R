### --------------------- Code for letter recognition project -------------------- ###

library(gbm)
library(ggplot2)
library(reshape2)  
library(gridExtra)
library(grid)
library(nnet)
library(kernlab)
library(randomForest)
library(MASS)

# Generate data set with B C D G O Q
letter <- read.csv("letter_recognition.csv", header = T)
attach(letter)
letterBCDGOQ <- letter[Class == "B"| Class == "C"| Class == "D"| Class == "G"|
                       Class == "O"| Class == "Q",]
write.csv(letterBCDGOQ, "letterBCDGOQ.csv")

letter6 <- read.csv("letterBCDGOQ.csv", header = T)[, -1]
letter.train <- letter6[1:3692,]
letter.test  <- letter6[-(1:3692),]

# Build gradient boosting model
set.seed(1234)
t <- proc.time()
gbm.letter <- gbm(Class ~ ., data = letter.train, distribution = "multinomial", n.trees = 10000,
                  shrinkage = 0.01, interaction.depth = 3, bag.fraction = 0.5, cv.folds = 10)
t1 <- proc.time() - t

# Check performance using 5-fold cross-validation
## black - train error; red - valid error; green - cv error
best.iter <- gbm.perf(gbm.letter, plot.it = F, method = "cv")

# Optimal iteration plot
df <- data.frame(iterarion = 1:10000, train = gbm.letter$train.error, cv = gbm.letter$cv.error) 
df_melt <- melt(df, id = "iterarion", variable.name = "method", value.name = "deviance")

theme <- theme(axis.title.y = element_text(vjust = 2, angle = 90, size = rel(0.78))) + 
         theme(axis.title.x = element_text(vjust = -1.2, angle = 00, size = rel(0.78))) +
         theme(plot.title = element_text(size = rel(0.9), face = "bold", vjust = 2.3)) +
         theme(axis.line = element_line(size=0.5), legend.title = element_blank(), 
               legend.key = element_blank())

opt.iter <- ggplot(data = df_melt, aes(x = iterarion, y = deviance, group = method)) + 
            geom_line(size = 0.4, aes(color = method, linetype = method)) + 
            geom_vline(xintercept = best.iter, linetype = 2) + 
            xlab("Iteration") + ylab("Multinomial Deviance") + theme +
            ggtitle("Figure 1    Train and cv error at each iteration")

# Relative importance
sum <- summary(gbm.letter, best.iter, plotit = F)

# Relative importance plot
inf_data <- data.frame(var = sum$var, inf = sum$rel.inf)
influ <- ggplot(inf_data, aes(x = reorder(var, inf), y = inf, fill = inf, width = 0.6)) +
        ## geom_bar() twice is a hack to omit slash from legend
        geom_bar(stat = "identity", position = "dodge") +
        scale_fill_continuous(low = "#56B1F7", high = "#132B43", space = "Lab") +
        scale_y_continuous(breaks = seq(0, 35, 5), limits = c(0, 35), expand = c(0, 0)) +
        labs(x = NULL, y = "Relative Influence") +
        coord_flip() + theme_bw() + 
        ggtitle("Figure 2    Relative importance of predictors") +
        theme(axis.text.x = element_text(size = rel(0.8)), 
              axis.text.y = element_text(size = rel(0.8)),                
              axis.title.x = element_text(vjust = -1.1, angle = 00, size = rel(0.78)),     
              panel.background = element_blank(), panel.border = element_blank(),
              legend.title = element_blank(), 
              legend.text = element_text(size = rel(0.77)),
              legend.key = element_blank(),
              plot.title = element_text(size = rel(0.8), face = "bold", vjust = 1.8))

# Marginal plots of xy2br (the most important predictor)
marg <- plot.gbm(gbm.letter, 12, best.iter, return.grid = T)
marg.xy2br <- marg[,1]

margB <- rep(0, nrow(marg))
margC <- rep(0, nrow(marg))
margD <- rep(0, nrow(marg))
margG <- rep(0, nrow(marg))
margO <- rep(0, nrow(marg))
margQ <- rep(0, nrow(marg))

for (i in 1:nrow(marg)) {
  margB[i] <- marg[i,2][1]
  margC[i] <- marg[i,2][2]
  margD[i] <- marg[i,2][3]
  margG[i] <- marg[i,2][4]
  margO[i] <- marg[i,2][5]
  margQ[i] <- marg[i,2][6]
}

margdf <- data.frame(xy2br = marg.xy2br, margB = margB, margC = margC, margD = margD,
                     margG = margG, margO = margO, margQ = margQ)
pB <- ggplot(margdf, aes(x = xy2br, y = margB)) + geom_line(colour = "#de2d26", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("B") + theme
pC <- ggplot(margdf, aes(x = xy2br, y = margC)) + geom_line(colour = "#9ebcda", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("C") + theme
pD <- ggplot(margdf, aes(x = xy2br, y = margD)) + geom_line(colour = "#fdae6b", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("D") + theme
pG <- ggplot(margdf, aes(x = xy2br, y = margG)) + geom_line(colour = "#c51b8a", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("G") + theme
pO <- ggplot(margdf, aes(x = xy2br, y = margO)) + geom_line(colour = "#a1d99b", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("O") + theme
pQ <- ggplot(margdf, aes(x = xy2br, y = margQ)) + geom_line(colour = "#c994c7", size = 0.82) + 
      labs(x = "xy2br", y = "Partial Dependence") + ggtitle("Q") + theme

## Arrange plots together
partial12 <- grid.arrange(pB, pC, pD, pG, pO, pQ, ncol = 3, nrow = 2, 
                          top = textGrob("Figure 3    Partial dependence of six letters on xy2br", 
                          gp = gpar(cex = 0.82, fontface = 2), vjust = -0.017))

# Prediction
pre.gbm <- predict(gbm.letter, newdata = letter.test, n.trees = best.iter, type = "response")
pre <- apply(pre.gbm, 1, which.max) ## BCDGOQ - 123456
pre[pre == 1] <- "B"
pre[pre == 2] <- "C"
pre[pre == 3] <- "D"
pre[pre == 4] <- "G"
pre[pre == 5] <- "O"
pre[pre == 6] <- "Q"
pre <- as.factor(pre)

true <- letter.test$Class
gbm.error <- 1 - sum(true == pre) / length(pre)

actual <- as.data.frame(table(true))
names(actual) <- c("Actual", "ActualFreq")

# Build confusion matrix
confusion <- as.data.frame(table(true, pre))
names(confusion) <- c("Actual", "Predicted", "Freq")

# Calculate percentage of test cases based on actual frequency
confusion <- merge(confusion, actual)
confusion$Percent <- confusion$Freq / confusion$ActualFreq * 100

# Plot confusion matrix
tile <- ggplot() + geom_tile(aes(x = Actual, y = Predicted, fill = Percent),
                             data = confusion, color = "white", size = 0.2) +
        labs(x = "Actual Class",y = "Predicted Class") + 
        geom_text(aes(x = Actual,y = Predicted, label = sprintf("%.1f", Percent)),
                  data = confusion, size=3, colour = "black") + 
        scale_fill_gradient(low = "#ece7f2", high = "#a6bddb")+  
        theme(panel.background = element_blank(), panel.grid.minor = element_blank(), 
              panel.grid.major = element_blank(), 
              axis.title.x = element_text(vjust = - 1.2, angle = 00, size = rel(0.78)), 
              axis.title.y = element_text(vjust = 2.1, angle = 90, size = rel(0.78)), 
              plot.title = element_text(size = rel(0.9), face = "bold", vjust = 1.8), 
              legend.title = element_text(size = rel(0.78)), 
              legend.text = element_text(size = rel(0.78))) + 
        ggtitle("Figure 4    Confusion matrix of the test set")

# Comparison with other classification methods
set.seed(1234)

## Multinomial logistic
t <- proc.time()
logit.letter <- multinom(Class ~ ., data = letter.train, ref = "B")
t2 <- proc.time() - t

pre.logit <- predict(logit.letter, newdata = letter.test, "probs")
pre.logit <- apply(pre.logit, 1, which.max)
true <- as.numeric(letter.test$Class)
logit.error <- 1 - sum(pre.logit == true) / length(pre.logit)

## Support vector machine
t <- proc.time()
svm.letter <- ksvm(Class ~ ., data = letter.train, type = "C-svc", 
                   kernel = "rbfdot", C = 3)
t3 <- proc.time() - t

true <- letter.test[,1]
pre.svm <- predict(svm.letter, letter.test[,-1])
svm.error <- 1 - sum(pre.svm == true) / length(pre.svm)

## Random forest
t <- proc.time()
rf.letter <- randomForest(Class ~ ., data = letter6, subset = 1:3692, ntree = 10000,
                          xtest = letter.test[,-1], ytest = letter.test[,1])
t4 <- proc.time() - t

pre.rf <- rf.letter$test$predicted
true <- letter.test[,1]
rf.error <- 1 - sum(pre.rf == true) / length(pre.rf)