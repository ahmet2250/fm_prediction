library(rvest)

options(max.print=1000000)

#/ Searching player's name

Player<- read.xlsx("Player.xlsx")

Player.name = Player$Name


result= 1:length(Player.name)


for (i in 1 : length(Player.name)) {
  
  Playerisim  <-  Player.name[i]
  x<-paste("https://www.google.co.in/search?q=", Playerisim , sep="")
  y<-paste(x, "+transfermarkt", sep="")
  ht <- read_html(y)
  links <- ht %>% html_nodes(xpath='//h3/a') %>% html_attr('href')
  url=gsub('/url\\?q=','',sapply(strsplit(links[as.vector(grep('url',links))],split='&'),'[',1))
  url[1]
  tryCatch({
  SearchURL <- read_html(url[1])

  PriceDatahtml <- html_nodes(SearchURL,'.dataMarktwert a')

  Pricedata <- html_text(PriceDatahtml)

  if(i>0){
    result[i] <- Pricedata } }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
  
  print(result[i]) 
 
}
write.csv(result, "scraped.csv")

