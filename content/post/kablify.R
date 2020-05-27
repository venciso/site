#Helper function to prettify html tables

kablify <- function(table){
        kable(table,format = 'html',format.args = list(big.mark = ',')) %>%
        kable_styling(bootstrap_options = 'striped')}
  
# kablify <- function(table,caption=""){
#   eval(parse(text=paste0("
#         kable(table,
#         format = 'html',format.args = list(big.mark = ','),
#         caption = '",caption,"') %>%
#         kable_styling(bootstrap_options = 'striped')"
#   )))}
  
#  kable(table,
#        format = 'html',format.args = list(big.mark = ",")) %>%
#    kable_styling(bootstrap_options = "striped")
