
library(stringr)

get_fnames_superA_range <- function(k, h) {
    Abegin <- (2**(h-1))+1
    Aend <- (2**h)-1
##    print(paste("Looking for A={",Abegin,"...",Aend,"}"))
    fnames <- list.files(path="../results",paste0("results_k",k,"_a"))
    if(length(fnames)==0){
        stop("Error, could not find files.")
    }
    for(f in fnames) {
        str_a <- str_match(f, "a([0-9]+)-([0-9]+)")
        if(length(str_a)<3)
            stop("could not match A values in filename.")
        a0 <- as.integer(str_a[,2])
        a1 <- as.integer(str_a[,3])
        if(a0<Abegin || a1>Aend)
            fnames <- fnames[fnames!=f]
    }
    if(length(fnames)<2) {
        stop("No super-A computation needed, aborting.")
    }
##    print(fnames)
    return(gtools::mixedsort(sort(fnames)))
}

get_superA <- function(k, h) {
    Abegin <- (2**(h-1))+1
    Aend <- (2**h)-1
    print(paste("Looking for A={",Abegin,"...",Aend,"}"))
    fnames <- list.files(path="../results",paste0("results_k",k,"_a"))
    if(length(fnames)==0){
        stop("Error, could not find files.")
    }
    for(f in fnames) {
        str_a <- str_match(f, "a([0-9]+)-([0-9]+)")
        if(length(str_a)<3)
            stop("could not match A values in filename.")
        a0 <- as.integer(str_a[,2])
        a1 <- as.integer(str_a[,3])
        if(!(Abegin==3 && a0==3 && Aend==3)
           && ( a0>Abegin || a1<=Abegin || a0>Aend || a1<Aend ))
            fnames <- fnames[fnames!=f]
        else {
            a0t <- a0
            a1t <- a1
        }
    }
    if(length(fnames)!=1) {
        stop("Found multiple files, aborting.")
    }

    x <- readLines(paste0("../results/",fnames[1]))
    line_id <- (Aend-a0t)/2 + 4 ## header rows
    s <- strsplit(x[line_id], ",")[[1]]
    c <- length(s)
    ## check by the way the two computed superAs (should always be equal)
    if( s[c] != s[c-3] )
        print("superA1 != superA2")
    minb <- s[c-2]
    mincb <- s[c-1]
    A <- s[c]
    
    ##   print(paste(minb,mincb,A))
    return(c(A,minb))

}

compute_superA_in_range <- function(k, h) {
    fnames <- get_fnames_superA_range(k, h)
    minb<-0
    mincb<-2**80
    A<-0
    for( f in fnames ) {
        x <- readLines(paste0("../results/",f))
        s <- strsplit(x[length(x)], ",")[[1]]
        c <- length(s)
        vminb <- as.integer(s[c-2])
        vmincb <- as.double(s[c-1])
        vA <- as.integer(s[c])
        ## check by the way the two computed superAs (should always be equal)
        if( vA != s[c-3] )
            print("superA1 != superA2")
        if( minb < vminb || (minb == vminb && mincb > vmincb)) {
            minb <- vminb
            mincb <- vmincb
            A <- vA
        }
    }
    ##   print(paste(minb,mincb,A))
    return(c(A,minb))
}


get_superAs <- function(ks=4:32, hs=2:16) {
    m <- as.data.frame(matrix(nrow=length(ks), ncol=length(hs),dimnames=list(paste0("k",ks),paste0("h",hs))))
    for(i in seq_along(ks)) {
        print(paste(i,"/",length(ks)))
        k<-ks[i]
        for(j in seq_along(hs)) {
            h<-hs[j]
            if(h+k>=43) {
                m[i,j] <- paste(paste(compute_superA_in_range(k,h),collapse=", "), "*")
            } else {
                m[i,j] <- paste(get_superA(k,h),collapse=", ")
            }
        }
    }    
    m
}

df <- get_superAs()
write.table(df,"superAs.csv",sep=",",dec=".")
