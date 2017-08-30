library(ggplot2)
library(shiny)
library(readr)
library(plyr)
library(dplyr)
library(scales)

## todo: use variable length of these arrays, omit *_0
## todo: t_kernel, t, sum .. are NA's (prolly too much NA's written) ...
## todo: ... fix counter in c++ code (p_i, hist_i)

open_csv <- function (i,fnames){
    fname<-paste0("results/",fnames[i])
    #extracting measurements
    local_frame <- read_csv(fname,skip=2,col_names=TRUE)
    colnames(local_frame) <- gsub(' ','_',colnames(local_frame))
   
    return(local_frame)
}

get_data_table <- function(params) {

    tab <- matrix(nrow=length(params$k)*length(params$A),
                  ncol=117)
    colnames(tab) <- c("id", "k", "A", "h", "A_2", "m_1", "m_2", "hlen", paste0("p_",1:50), paste0("hist_",1:50), "t_kernel", "t", "sum", "minb", "mincb", "superA", "minb2", "mincb2", "superA2")
    i=1
    for(k in params$k) {
        fname<-paste0("results_k", k, "_a")
        fnames <- list.files(path="results",fname)
        if(length(fnames)==0){
            stop(paste0("Error, could not find files (",fname,"..)."))
        }
        for(A in params$A) {
            for(f in fnames) {
                str_a <- stringr::str_match(f, "a([0-9]+)-([0-9]+)")
                if(length(str_a)<3)
                    stop("could not match A values in filename.")
                a0 <- as.integer(str_a[,2])
                a1 <- as.integer(str_a[,3])
                if( a0<=A && A<=a1 ) {
                    ## if(length(fnames)==0) {
                    ##     stop("Could not find files, aborting.")
                    ## }
                    ## if(length(fnames)!=1) {
                    ##     stop("Found multiple files, aborting.")
                    ## }

                    x <- readLines(paste0("results/",f))
                    line_id <- (A-a0)/2 + 4 ## header rows
                    s <- strsplit(x[line_id], ",")[[1]]
                    c <- k + floor(log2(A))

                    tab[i,1:8] <- as.integer(s[1:8])
                    tab[i,9:(9+c+1)] <- as.double(c(s[9:(9+c)],0)) ## p
                    tab[i,59:(59+c+1)] <- as.double(c(s[(10+c):(10+2*c)],0)) ## hist
                    tab[i,109:117] <- as.double(s[(11+2*c):length(s)]) ## 
                    i=i+1
                }
            }
        }
    }
    return(as.data.frame(tab))
}

get_params_default <- function() {
    params <- list()
    params$title <- ""
    params$plot <- 'probability'
    params$k <- c(4)
    params$A <- c(3)
    return(params)
}

get_params <- function(input) {

    params <- get_params_default()
    params$title <- input$sTitle
    params$k <- as.integer(strsplit(input$sK,",")[[1]])
    params$A <- as.integer(strsplit(input$sA,",")[[1]])
    params$k <- params$k[ params$k>3 & params$k<33 ]
    params$A <- params$A[ params$A>2 & params$A<65536 ]
    params$A <- params$A[ (params$A %% 2)==1 ]
    params$plot = input$sPlotType

    if(length(input$sTableA_cells_selected)) {
        indices <- input$sTableA_cells_selected
        params$A <- as.numeric(superA$table[ indices ])
        params$k <- as.numeric(indices[,1])+3
    }
    params$k <- as.vector(unique(params$k))
    params$A <- as.vector(unique(params$A))
    return(params)
}

###
## --- View ---
###

get_superAs_table <- function(html=FALSE) {
    tab <- read.csv("results/superAs.csv",stringsAsFactors=FALSE)
    rownames(tab) <- gsub("k","k=",rownames(tab))
    colnames(tab) <- gsub("h","h=",colnames(tab))
    for(i in 1:nrow(tab)) {
        for(j in 1:ncol(tab)) {
            v <- strsplit(toString(tab[i,j]), ", ")
            if(html)
                tab[i,j]<-paste0("<strong>",v[[1]][1],"</strong> <em>(",v[[1]][2],")</em>")
            else
                tab[i,j]<-as.numeric(v[[1]][1])
        }
    }
    return(tab)
}

superA <- list()
superA$table <- get_superAs_table()
superA$table_html <- get_superAs_table(html=TRUE)

## id, k, A, h, A_2, m_1, m_2, p_[50], hist_[50], t_kernel, t, sum, minb, mincb
## 
## plots: histograms (p,hist) of a selected (k,A) (comparison of 2) ((VIEW 2))
## tables: superAs (k,h) (--link-to-view_2--, update gui param)  ((MAIN VIEW))
## plots: minb+mincb of a selected (k,h) (y2-plot (mincb, min)) ((MAIN SUBVIEW)) ## worst-case 16k axis points (uniq on mincb)

## superA: read min* of last entry of h-class

get_tables <- function(data, params) {

    tables <- list()

    cmax <- max(unlist(data$hlen))-1
    if(params$plot == "probability") {
        succeeded_y  <- stack(data[,9:(8+cmax)])
        tables$ylabel <- "SDC-probability p"
    }
    else {
        succeeded_y  <- stack(data[,59:(58+cmax)])
        tables$ylabel <- "Frequency c_b"
    }
    ## aes by 'k,A'
    nk <- length(params$k)
    nA <- length(params$A)
    nkA <- nk*nA
    ## (k,A) -> (p_1..p_50)
    ## 'k 4,..,4,|,4,..,4
    ##  A 3,..,3,|,5,..,5
    ##  p p_1,..,p_50,|,p_1,..,p_50 '
    ex_k <- rep(unlist(lapply(params$k,rep,times=nA)),cmax)
    ex_A <- rep(params$A,cmax)
    dp <- data.frame(x=floor(seq(0,nkA*cmax-1)/nkA)+1, y=succeeded_y, A=as.factor(ex_A), k=as.factor(ex_k))
    ## 2^k * choose(k+h, x)
    ex_h <- floor(log2(ex_A))+1
    
    dp$base <- 2**ex_k * choose(ex_k+ex_h, dp$x)
    tables$raw <- data
    tables$reduced <- dp
    tables$title <- params$title

    return(tables)
}

plot_data <- function(tables,
                      usepoints=T,
                      nolegend=F,
#                      freqpoly=F,
#                      bins=200,
                      xlimit="",
                      ylimit="",
                      logx="-",
                      logy="-") {
##    succeeded_reduced <- tables$raw
    data_for_plotting <- tables$reduced
    xlabel <- 'bitflips b'    
    ylabel <- tables$ylabel
  
    my_theme <-  theme_bw() + theme(axis.title.x = element_text(size=18),
                                    axis.title.y = element_text(size=18),
                                    axis.text.x = element_text(size=14),
                                    axis.text.y = element_text(size=14)#,
                                        #axis.text.x  = element_text()
                                   ,plot.margin = unit(c(8,10,1,1), "pt") # required otherwise labels are clipped in pdf output
                                    )
    my_theme <- my_theme + theme(legend.title = element_text(size=16, face="bold"),#legend.title = element_blank()
                                 legend.text = element_text( size = 16),
                                 legend.position="bottom",
                                 legend.direction="vertical",
                                 legend.box ="horizontal",
                                 legend.box.just ="bottom",
                                 legend.background = element_rect(colour = 'white', fill = 'white', size = 0., linetype='dashed'),
                                 legend.key = element_rect(colour = 'white', fill = 'white', size = 0., linetype='dashed'),
                                 legend.key.width = unit(1.1, "cm")
                                 )


    moi_plot <- ggplot(data_for_plotting, aes(x=x))
    moi_plot <- moi_plot + geom_line(aes(y=y.values,colour=A,linetype=k),size=1)
    if( usepoints ) {
        moi_plot <- moi_plot + geom_point(aes(y=y.values),size=3)
    }
##    moi_plot <- moi_plot + scale_linetype_manual(values = c("solid","dotted","longdash"))
    moi_plot <- moi_plot + scale_x_discrete(limits=1:max(data_for_plotting$x))

    ##

    moi_plot <- moi_plot + ylab(ylabel) + xlab(xlabel)
    moi_plot <- moi_plot + my_theme

    if(nchar(tables$title)>1)
        moi_plot <- moi_plot + ggtitle(tables$title)

    str_to_numeric = function( string, sep ) {

        splitted = unlist(strsplit(string,sep))
        vec = sapply(splitted, function(x) as.numeric(x))
        return(vec)
    }

    ## ylimit_splitted = unlist(strsplit(opts[["ylimit"]],","))
    ## ylimit_pair = sapply(ylimit_splitted, function(x) as.numeric(x))
    ylimit_pair = str_to_numeric(ylimit, ",")
    xlimit_pair = str_to_numeric(xlimit, ",")

    if( length(ylimit_pair) == 2 ) {
        if(ylimit_pair[1] != 0 || ylimit_pair[2]!=0){
            cat("[ylimit] setting to ",paste(ylimit_pair),"\n")
            moi_plot <- moi_plot + ylim(ylimit_pair[1],ylimit_pair[2])
        }
    }

    if( length(xlimit_pair) == 2 ) {
        if(xlimit_pair[1] != 0 || xlimit_pair[2]!=0){
            cat("[xlimit] setting to ",paste(xlimit_pair),"\n")
            moi_plot <- moi_plot + xlim(xlimit_pair[1],xlimit_pair[2])
        }
    }


    if(nolegend){
        moi_plot <- moi_plot + theme(legend.position="none")
    }

    logx_value <- 1
    logy_value <- 1
    if(logx!="-")
        logx_value <- as.integer(logx)
    if(logy!="-")
        logy_value <- as.integer(logy)

    xmin <- min(data_for_plotting$x)
    xmax <- max(data_for_plotting$x)
## todo: hist
    ymin <- min(data_for_plotting$y.values)
    ymax <- max(data_for_plotting$y.values)


    if(logy_value > 1) {

        breaks_y = function(x) logy_value^x
        format_expr_y = eval(parse(text=paste("math_format(",logy_value,"^.x)",sep="")))

        if(length(ylimit_pair) == 2 && (ylimit_pair[1] != 0 && ylimit_pair[2]!=0)){
            scale_structure = scale_y_continuous(
                limits = ylimit_pair,
                trans = log_trans(base=logy_value),
                breaks = trans_breaks(paste("log",logy_value,sep=""), breaks_y),
                labels = trans_format(paste("log",logy_value,sep=""), format_expr_y))
        } else {
            scale_structure = scale_y_continuous(
                trans = log_trans(base=logy_value),
                breaks = trans_breaks(paste("log",logy_value,sep=""), breaks_y),
                labels = trans_format(paste("log",logy_value,sep=""), format_expr_y))
            
        }

        moi_plot <- moi_plot + scale_structure
    }



    if(logx_value > 1) {

        breaks_x = function(x) logx_value^x
        format_expr_x = eval(parse(text=paste("math_format(",logx_value,"^.x)",sep="")))
        if(length(xlimit_pair) == 2 && (xlimit_pair[1] != 0 && xlimit_pair[2]!=0)){
            scale_x_structure = scale_x_continuous(
                limits = xlimit_pair,
                trans = log_trans(base=logx_value),
                breaks = trans_breaks(paste("log",logx_value,sep=""), breaks_x),
                labels = trans_format(paste("log",logx_value,sep=""), format_expr_x)
            )
        } else {
            scale_x_structure = scale_x_continuous(
                trans = log_trans(base=logx_value),
                breaks = trans_breaks(paste("log",logx_value,sep=""), breaks_x),
                labels = trans_format(paste("log",logx_value,sep=""), format_expr_x)
            )
            
        }

        moi_plot <- moi_plot + scale_x_structure
    }

    
    return(moi_plot)
}

##
## -- SERVER --
##


server <- function(input, output, session) {

    output$sTable <- DT::renderDataTable(DT::datatable({

        params <- get_params(input)

        df_data <- get_data_table(params)
        tables <- get_tables(df_data, params)

        return(tables$reduced)
    }))

    output$sTableA <- DT::renderDataTable(DT::datatable({superA$table_html},
                                                        escape=F,
                                                        selection = list(target = 'cell'),
                                                        options = list(
                                                            lengthMenu = list(c(5, 10, 20, -1), c('5', '10', '20', 'All')),
                                                            pageLength = 5
                                                        )))
    
    output$sPlot <- renderPlot({

        params <- get_params(input)

        df_data <- get_data_table(params)
        tables <- get_tables(df_data, params)


        plot_data(tables,
                  usepoints=input$sUsepoints,
                  logx = input$sLogx,
                  logy = input$sLogy,
                  xlimit = input$sXlimit
                  )
    })

}





## ---------------
## User Interface
## ---------------

ui <- fluidPage(

    theme="simplex.min.css",
    tags$style(type="text/css",
               "label {font-size: 12px;}",
               "p {font-weight: bold;}",
               "h3 {margin-top: 0px;}",
               ".checkbox {vertical-align: top; margin-top: 0px; padding-top: 0px;}",
               "table.dataTable.hover tbody tr td:hover, table.dataTable.display tbody tr td:hover {background-color: #ffff99 !important; cursor:pointer;}"
               ),

    h1("AN-Coding Analysis Tool"),
    p("Compare for different integer values A and data word widths k the silent data corruption probabilities. Get ",
      a(href="https://github.com/tuddbresilience/coding_reliability", "source code on github"),
      " (/distance_distribution_super_a/)."),
    p("Values have been computed for k={4,...,32} and A={3,5,...,65535}, where the bigger problems have been approximated by a grid sampling method."),
    hr(),

    wellPanel(
        h3("Parameters"),
        fluidRow(
            column(3, textInput("sK", "data word width k in {4,5,...,32}", "4,5")),
            column(3, textInput("sA", "A in {3,5,...,65535}", "5,9")),
            column(3, textInput("sTitle", "Plot Title", ""))
        )
    ),
    wellPanel(
        div(DT::dataTableOutput("sTableA"), style = "font-size: 75%;"),
        br(),
        wellPanel(
            h3("Plot Options"),
            fluidRow(
                column(3, selectInput("sPlotType", "Plot type", c("probability","counts"), selected="probability")),
                column(2, selectInput("sLogx", "Log-X", c("-","2","10"), selected="-")),
                column(2, selectInput("sLogy", "Log-Y", c("-","2","10"), selected="-")),
                column(2, textInput("sXlimit", "x-range", "0,0")),
                column(2, checkboxInput("sUsepoints", "Draw Points"))
                
            )
        ),
        br(),
        plotOutput("sPlot"),
        br(),
        DT::dataTableOutput("sTable")
    ),
    hr(),
    
    ## fluidRow(verbatimTextOutput("log"))
    ##    mainPanel(plotOutput("distPlot"))
    ##  )
    
    span("This tool is powered by R Shiny Server.")
)

## will look for ui.R and server.R when reloading browser page, so you have to run
## R -e "shiny::runApp('~/shinyapp')"
shinyApp(ui = ui, server = server)
