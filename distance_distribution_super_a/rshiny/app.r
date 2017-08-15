
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

## this binds the data frames together
get_data <- function(fnames) {
    result <- ldply(seq_along(fnames), .fun=open_csv, fnames=fnames)
    return(result)
}


###
## --- View ---
###


#flist <- list.files(pattern = ".csv$", recursive = TRUE)

get_input_files <- function(args) {

    fname<-paste0("results_n",args$k, "_a3-65535.csv")
    return(fname)
}

get_args_default <- function() {
    args <- list()
    args$k <- 8
    args$h <- 8
    return(args)
}

get_args <- function(input) {

    args <- get_args_default()
    args$k <- input$sK
    args$h <- input$sh
    return(args)

}

## id, k, A, h, A_2, m_1, m_2, p_[50], hist_[50], t_kernel, t, sum, minb, mincb
## 
## plots: histograms (p,hist) of a selected (k,A) (comparison of 2) ((VIEW 2))
## tables: superAs (k,h) (--link-to-view_2--, update gui param)  ((MAIN VIEW))
## plots: minb+mincb of a selected (k,h) (y2-plot (mincb, min)) ((MAIN SUBVIEW)) ## worst-case 16k axis points (uniq on mincb)

## superA: read min* of last entry of h-class

get_tables <- function(data, args) {
    ## filter_mode <- ""
    ## filter_prec <- ""
    ## filter_type <- ""
    ## filter_kind <- ""
    ## filter_dim  <- 0

    if(nchar(args$run)>0 && args$run == "-")
        filter_run <- c("Warmup", "Success")
    else
        filter_run <- args$run


    filtered_by <- c("success")

    if(nchar(args$inplace)>1){
        filter_mode <- args$inplace
    }

    if(nchar(args$precision)>1){
        filter_prec <- args$precision
    }

    if(nchar(args$complex)>1){
        filter_type <- args$complex
    }

    if(nchar(args$kind)>1){
        filter_kind <- args$kind
    }

    if(nchar(args$dim)>0 && args$dim!='-') {
        filter_dim <- as.integer(args$dim)
    }

    if(args$xmetric=='id')
        xlabel <- 'id'
    else if(args$xmetric=="nbytes")
        xlabel <- "Signal_Size_[bytes]"
    else
        xlabel <- args$xmetric
    
    if(grepl("Time", args$xmetric))
        xlabel <- paste0(args$xmetric,"_[ms]")

    if(grepl("/", args$ymetric))
        ylabel <- paste0(args$ymetric,"_[%]")
    else if(grepl("Time", args$ymetric))
        ylabel <- paste0(args$ymetric,"_[ms]")
    else if(grepl("Size", args$ymetric))
        ylabel <- paste0(args$ymetric,"_[bytes]")

    succeeded <- data ### 

    if ( nchar(filter_mode) > 0){
        succeeded <- succeeded %>% filter(inplace == filter_mode)
        cat("filtered for inplace == ",filter_mode,": \t",nrow(succeeded),"\n")
        filtered_by <- c(filtered_by, filter_mode)
    }

    if ( nchar(filter_type) > 0 ){
        succeeded <- succeeded %>% filter(complex == filter_type)
        cat("filtered for complex == ",filter_type,": \t",nrow(succeeded),"\n")
        filtered_by <- c(filtered_by, filter_type)
    }

    if ( nchar(filter_prec) > 0){
        succeeded <- succeeded %>% filter(precision == filter_prec)
        cat("filtered for precision == ",filter_prec,": \t",nrow(succeeded),"\n")
        filtered_by <- c(filtered_by, filter_prec)
    }

    if ( nchar(filter_kind) > 0 && !("all" %in% filter_kind) ){
        succeeded <- succeeded %>% filter(kind == filter_kind)
        cat("filtered for kind == ",filter_kind,": \t",nrow(succeeded),"\n")
        filtered_by <- c(filtered_by, filter_kind)
    }

    if ( filter_dim > 0){
        succeeded <- succeeded %>% filter(dim == filter_dim)
        cat("filtered for ndims == ",filter_dim,": \t",nrow(succeeded),"\n")
        filtered_by <- c(filtered_by, paste(filter_dim,"D",sep=""))
    }
##############################################################################
    data_colnames = colnames(succeeded)

                                        # extracting ymetric expression
    ymetric_keywords = trimws(unlist(strsplit(args$ymetric,"[-|+|/|*|)|(]")))
    ymetric_expression = args$ymetric

                                        # creating expression
    for(i in 1:length(ymetric_keywords)) {

        indices = grep(ymetric_keywords[i],data_colnames)
        if( length(indices) > 0 && !is.null(ymetric_keywords[i]) && nchar(ymetric_keywords[i]) > 1){
            to_replace = paste("succeeded[,",indices[1],"]",sep="")
            cat(i,ymetric_keywords[i],"->",to_replace,"in",ymetric_expression,"\n")
            ymetric_expression = gsub(ymetric_keywords[i],to_replace,
                                      ymetric_expression)
        }
    }


                                        # creating metric of interest (moi)
    new_values = as.data.frame(eval(parse(text=ymetric_expression)))
    colnames(new_values) <- c("ymoi")

    name_of_ymetric = args$ymetric

    if( length(ymetric_keywords) == 1  ){
        name_of_ymetric = data_colnames[grep(ymetric_keywords[1], data_colnames)[1]]
    }

    if(!is.null(ylabel)) {

        if( nchar(ylabel) > 1){
            name_of_ymetric = gsub("_"," ",ylabel)
        }
    }
    cat("[ylabel] using ylabel: ",name_of_ymetric,"\n")

    succeeded_ymetric_of_interest  = new_values
################################################################################

##############################################################################
                                        # extracting xmetric expression
    if(grep(args$xmetric,data_colnames) == 0){

        stop(paste(args$xmetric, "for x not found in available columns \n",data_colnames,"\n"))
    }


    succeeded_xmetric_of_interest  <- succeeded %>% select(contains(args$xmetric))
    name_of_xmetric <- colnames(succeeded_xmetric_of_interest)[1]
    if(!is.null(xlabel)) {

        if( nchar(xlabel) > 1){
            name_of_xmetric = xlabel
        }
    }
    colnames(succeeded_xmetric_of_interest) <- c("xmoi")
    succeeded_factors <- succeeded %>% select(-ends_with("]"))

    succeeded_reduced <- bind_cols(succeeded_factors,
                                   succeeded_xmetric_of_interest,
                                   succeeded_ymetric_of_interest)

    if( grepl("bytes",name_of_xmetric)  ) {
        succeeded_reduced$xmoi <- succeeded_reduced$xmoi / (1024.*1024.)
        name_of_xmetric <- gsub("bytes","MiB",name_of_xmetric)
    }

    if( grepl("bytes",name_of_ymetric) ){
        succeeded_reduced$ymoi <- succeeded_reduced$ymoi / (1024*1024)
        name_of_ymetric <- gsub("bytes","MiB",name_of_ymetric)
    }


    cols_to_consider <- Filter(function(i){ !(i %in% filtered_by || i == "id" || i == "run") },c(colnames(succeeded_factors),"xmoi"))
    cols_to_grp_by <- lapply(c(cols_to_consider,"id"), as.symbol)

    data_for_plotting <- succeeded_reduced %>%
        group_by_(.dots = cols_to_grp_by) %>%
        ##group_by(library, hardware, id, nx, ny, nz, xmoi) %>%
        summarize( moi_mean = mean(ymoi),
                  moi_median = median(ymoi),
                  moi_stddev = sd(ymoi)
                  )

    tables <- list()
    tables$raw <- succeeded_reduced
    tables$reduced <- data_for_plotting
    tables$name_of_xmetric <- name_of_xmetric
    tables$name_of_ymetric <- name_of_ymetric
    if( args$notitle == F ) {
        tables$title <- paste("filtered by:",paste(filtered_by,collapse=" "))
    } else {
        tables$title <- ""
    }

#    tables <- data_for_plotting[c('id','xmoi','moi_mean','moi_median','moi_stddev')]
    return(tables)
}


server <- function(input, output, session) {
    
    output$sPlot <- renderPlot({

        args <- get_args(input)
        ## if(is.null(input$file1)) {
        ##     return()
        ## }
        input_files <- get_input_files(args)

        df_data <- get_data(input_files)
        tables <- get_tables(df_data, args)

        ## aesthetics
        aes <- c()
        if(input$sAes!="-")
            aes <- append(aes,input$sAes)
        aes_str <- paste(aes, collapse=",")

        freqpoly <- F
        usepointsraw <- F
        usepoints <- F
        noerrorbar <- F

        ## plot type
        if(input$sPlotType=="Histogram") {
            freqpoly <- T
            noerrorbar <- T
        } else if(input$sPlotType=="Points") {
            usepointsraw <- T
        } else {
            usepoints <- input$sUsepoints || length(aes)>2
            noerrorbar <- input$sNoerrorbar
        }

        plot_gearshifft(tables,
                        aesthetics = aes_str,
                        logx = input$sLogx,
                        logy = input$sLogy,
                        freqpoly = freqpoly,
                        bins = input$sHistBins,
                        usepoints = usepoints,
                        usepointsraw = usepointsraw,
                        noerrorbar = noerrorbar
                        )
    })

    output$sPlotOptions <- renderUI({
        if(input$sPlotType == "Histogram")
            column(2, numericInput("sHistBins", "Bins", 200, min=10, max=1000))
        else if(input$sPlotType == "Lines") {
            fluidRow(column(1, checkboxInput("sUsepoints", "Draw Points")),
                     column(2, checkboxInput("sNoerrorbar", "Disable Error-Bars")))
        }
    })

    output$sInfo <- renderUI({
        input_files <- get_input_files(input)
        header <- get_gearshifft_header( input_files[1] )
        output$table1 <- renderTable({
            header$table1
        })
        output$table2 <- renderTable({
            header$table2
        })

        if(length(input_files)>1) {
            header2 <- get_gearshifft_header( input_files[2] )
            output$table3 <- renderTable({
                header2$table1
            })
            output$table4 <- renderTable({
                header2$table2
            })
            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(
                    column(4, tableOutput("table1")),
                    column(4, tableOutput("table2"))
                ),
                h4(input_files[2]),
                fluidRow(
                    column(4, tableOutput("table3")),
                    column(4, tableOutput("table4"))
                )
            )
        } else {

            wellPanel(
                br(),
                h4(input_files[1]),
                fluidRow(
                    column(4, tableOutput("table1")),
                    column(4, tableOutput("table2"))
                )
            )
        }
    })

    #
    output$sHint <- renderUI({
        if(input$sPlotType == "Histogram")
            p("Histograms help to analyze data of the validation code.", HTML("<ul><li>Use Time_* as xmetric for the x axis.</li><li>Probably better to disable log-scaling</li><li>If you do not see any curves then disable some filters.</li></ul>"))
        else if(input$sPlotType == "Lines")
            p("Lines are drawn by the averages including error bars.", HTML("<ul><li>If you see jumps then you should enable more filters or use the 'Inspect' option.</li><li>Points are always drawn when the degree of freedom in the diagram is greater than 2.</li></ul>"))
        else if(input$sPlotType == "Points")
            p("This plot type allows to analyze the raw data by plotting each measure point. It helps analyzing the results of the validation code.")

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
               ".checkbox {vertical-align: top; margin-top: 0px; padding-top: 0px;}"
               ),

    h1("AN-Coding Analysis Tool"),
#    p("gearshifft is an FFT benchmark suite to evaluate the performance of various FFT libraries on different architectures. Get ",
#      a(href="https://github.com/mpicbg-scicomp/gearshifft/", "gearshifft on github.")),
    hr(),

    wellPanel(
        h3("Parameters"),
        br(),
        fluidRow(
            column(2, selectInput("sK", "data word width (k)", seq(4,8), selected="8")),
            column(2, selectInput("sh", "width A (h)", seq(2,16), selected="8")),
            column(2, selectInput("sAes", "Inspect", c("-","param_k","param_h"), selected="-"))
        )
    ),

    tabsetPanel(
        ## Plot panel
        tabPanel("Plot",

                 br(),
                 plotOutput("sPlot"),
                 br(),
                 wellPanel(
                     h3("Plot Options"),
                     fluidRow(
                         column(3, selectInput("sPlotType", "Plot type", c("Lines","Histogram","Points"), selected="Lines")),
                         column(1, selectInput("sLogx", "Log-X", c("-","2","10"), selected="2")),
                         column(1, selectInput("sLogy", "Log-Y", c("-","2","10"), selected="10")),
                         column(1, checkboxInput("sNotitle", "Disable Title")),
                         uiOutput("sPlotOptions")
                     ),
                     uiOutput("sHint"))),
        tabPanel("Info",                 
                 br(),
                 uiOutput("sInfo")
                 )
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
