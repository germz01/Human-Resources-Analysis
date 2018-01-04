library(ggplot2)

df <- read.csv("df_general.csv")
head(df)

# grafico regole generali

p<-ggplot(df, aes(x=Lift,y=Conf,
                  fill=Consequent_,
                  col=Consequent_,
                  size=Supp))
p<-p+geom_point(alpha=0.3)
p<-p+geom_smooth(method="lm",size=1)#,col="black")
p<-p+geom_vline(xintercept = 1,col="red",size=3,linetype="solid")
p<-p+ggtitle("Association Rules for MinSupp=5%")
p<-p+scale_size(range=c(10,20),trans="identity")
p<-p+scale_color_brewer(palette='Paired')
p<-p+scale_fill_brewer(palette='Paired')
p<-p+guides(colour=FALSE,fill=FALSE)
p<-p+guides(colour = guide_legend(order = 2),
            fill = guide_legend(order = 2)
            )
p+theme_bw()
                                        #shape = guide_legend(order = 2))

ggsave("../../images/rules/rules_general.pdf",width=10,height=7)

###########################################################
library(ggplot2)

df <- read.csv("df_specific.csv")
head(df)

# grafico regole specifici

p<-ggplot(df, aes(x=Lift,y=Conf,
                  fill=Consequent_,
                  col=Consequent_,
                  size=Supp))
p<-p+geom_point(alpha=0.3)
p<-p+geom_smooth(method="lm",size=1)#,col="black")
p<-p+geom_vline(xintercept = 1,col="red",size=3,linetype="solid")
p<-p+ggtitle("Association Rules for MinSupp=5%")
p<-p+scale_size(range=c(10,20),trans="identity")
p<-p+scale_color_brewer(palette='Paired')
p<-p+scale_fill_brewer(palette='Paired')
p<-p+guides(colour=FALSE,fill=FALSE)
p<-p+guides(colour = guide_legend(order = 2),
            fill = guide_legend(order = 2)
            )
p+theme_bw()
                                        #shape = guide_legend(order = 2))

ggsave("../../images/rules/rules_specific.pdf",width=10,height=7)

###########################################################




