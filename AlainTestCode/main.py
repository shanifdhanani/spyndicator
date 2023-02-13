
import stockQuote
import functions_framework
import json

@functions_framework.http

def get_quote(request):

    x = stockQuote.myQuote("SPX", "1min", 3)
   
    j = x.getQuote()
    print(j)   
    j_text = json.dumps(j)
    #j_text = "Hi"
    sample_text = "THIS IS THE LATEST SPX DATA "
    return_text = sample_text + j_text
    return return_text

#print(get_quote("test"))