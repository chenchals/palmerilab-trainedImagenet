rm(list=ls())

load_file = function(file_name){
     load(file_name); 
     obj_names = ls()
     obj_names = obj_names[obj_names != 'file_name']
     return(get(obj_names))
}

load_theme = function(){
     require(ggplot2)
     mytheme = theme(axis.line = element_line(colour = "black"),
                     axis.text = element_text(colour = "black"),
                     panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(),
                     strip.background = element_blank(),
                     panel.border = element_rect(color='black',fill=NA),
                     panel.background = element_blank())  
     return(mytheme)
}

plot_mds = function(data){
     require(GGally)
     require(smacof)
     require(plyr)
     
     
     
     mytheme = load_theme()
     
     
     lower_facet =  function(data, mapping, ...) {
          p = ggplot(data = data, mapping = mapping) +
               geom_point() +
               geom_line() + 
               mytheme
     } 
     p1 = ggpairs(mds,columns=cols,mapping=aes(color=Category,shape=Sex,group=ID_2),
                  lower=list(continuous=lower_facet),
                  axisLabels='internal',
                  columnLabels = plot_labs)
     
     
     
     points_legend = gglegend(ggally_points)
     x = points_legend(mds,aes(x=Dimension1,y=Dimension2,color=Category,shape=Sex))
     
     n_dims = length(plot_labs)
     ut = which(upper.tri(matrix(NA,n_dims,n_dims)),arr.ind = T)
     for(i in 1:nrow(ut)){
          p1[ut[i,1],ut[i,2]] = NULL
     }
     p1[2,n_dims] = x
     
     print(p1)
     return(p1)
}


plot_gof = function(df,num_dims=NULL){
     require(ggplot2)
     mytheme = load_theme()
     if(is.null(num_dims))num_dims = length(unique(df$Dimension))
     df = df[df$Dimension <= num_dims, ]
     p1 = ggplot(df,aes(x=Dimension,y=Stress,group=Viewpoint,color=Viewpoint)) +
          geom_line() +
          geom_point() +
          mytheme
     plot(p1)
     return(p1)
}


library(plyr)
library(plotly)
library(smacof)

#########################

wd = '~/Projects/lab-palmeri/trainedImagenet/JeffsRCode/mds_plot_chenchal/net_model/' #working directory
dst_file = '~/Projects/lab-palmeri/trainedImagenet/JeffsRCode/mds_plot_chenchal/data/distance_euc.RData' #distance matrix
ftr_file = '~/Projects/lab-palmeri/trainedImagenet/JeffsRCode/mds_plot_chenchal/data/feature_names.RData' #feature names
plot_file = '~/Projects/lab-palmeri/trainedImagenet/JeffsRCode/mds_plot_chenchal/plots/all_mds_3_ggally.eps' #file name of plot to save
type = 'Greeble 1' #choose 1: 'Greeble 1','Greeble 2','Ziggerin 1','Ziggerin 2','Sheinbug'
k = 3 #MDS dimensions

#########################

setwd(wd)

img_idx = unlist(mapvalues(type,
                    c('Greeble 1','Greeble 2','Ziggerin 1','Ziggerin 2','Sheinbug'),
                    list(1:100,101:200,201:300,301:400,401:500),
                    warn_missing = FALSE))

dst = load_file(dst_file)
ftr_names = load_file(ftr_file)   

mds = smacofSym(delta = dst[img_idx,img_idx],ndim = k)$conf

cols = paste0('Dimension',1:ncol(mds))
colnames(mds) = cols
plot_labs = paste('Dimension',1:ncol(mds))
ftr_names$ID_2 = rep(seq(1:(nrow(ftr_names)/2)),each=2)
ftr_names$Category = mapvalues(ftr_names$Category,c('ng1','ng2','nz1','nz2','sb'),
                               c('Greeble 1','Greeble 2','Ziggerin 1','Ziggerin 2','Sheinbug'))
ftr_names$Sex = mapvalues(ftr_names$Sex,c('f','m','NA'),c('Female','Male','NA'))
ftr_names = ftr_names[img_idx,]
mds = cbind(mds,ftr_names)

plot_mds(mds)
plot_ly(z=as.matrix(dst),type='heatmap')
#ggsave(plot_file)




