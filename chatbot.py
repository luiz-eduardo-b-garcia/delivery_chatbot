#importa numpy para manipulação dos dados em matrizes
import numpy as np
!pip install python-telegram-bot --upgrade
from telegram import Update, ForceReply
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
#acessa o drive
from google.colab import drive
drive.mount('/content/drive')

#acessa o dataset
from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

worksheet = gc.open('acaiteria').sheet1

#pega os dados em linhas
rows = worksheet.get_all_values()
print(rows)

#transforma em um dataframe
import pandas as pd
df=pd.DataFrame(rows[1:],columns=rows[0])

#cria dataframe para treino
intencoes = ['iniciar_conversa','pedir_cardapio','fazer_pedido', 'fechar_conta', 'passar_endereco', 'tempo_entrega', 'situacao_pedido']
xtrain_global = []
ytrain_global = []
for intencao in intencoes:
    lintencao = df[df['Conjunto']=='Treino'][intencao].values.tolist()
    xtrain_global += lintencao
    ytrain_global += [intencao]*len(lintencao)


#cria dataframe para teste
xtest = []
ytest = []
for intencao in intencoes:
    lintencao = df[df['Conjunto']=='Teste'][intencao].values.tolist()
    xtest += lintencao
    ytest += [intencao]*len(lintencao)

#confere se o dataset de treino condiz com a planilha, tendo 61 linhas e 7 colunas de dados, 61*7= 427
len(xtrain_global)

#confere se o dataset de treino condiz com a planilha, tendo 19 linhas e 7 colunas de dados, 19*7= 133
len(xtest)

#inicia processamento de dados
!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
converter = SentenceTransformer('multi-qa-distilbert-cos-v1')

xtrain_emb = converter.encode(xtrain_global)
xtest_emb = converter.encode(xtest)

#utiliza leave one out para induzir o modelo, optei por esse metodo devido a pequena quantidade de amostras que possuia
import sklearn.neighbors as neighbors
from sklearn.model_selection import LeaveOneOut
model = neighbors.KNeighborsClassifier()
loo = LeaveOneOut()
loo.get_n_splits(xtrain_emb)
ytrain_array = np.array(ytrain_global)

res = []
ytrue = []
ypred = []
for i,(train_index, test_index) in enumerate(loo.split(xtrain_emb)):
    model.fit(xtrain_emb[train_index],ytrain_array[train_index])
    pred = model.predict(xtrain_emb[test_index])
    ytrue.append(ytrain_array[test_index])
    ypred.append(pred)

#mostra as metricas do modelo induzido para controle
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

print(metrics.classification_report(ytrue,ypred))

from sklearn import model_selection

def evaluate_pred(y_true,y_pred):
    prec   = precision_score(y_true,y_pred,average='macro')
    recall = recall_score(y_true,y_pred,average='macro')
    f1     = f1_score(y_true,y_pred,average='macro')
    acc    = accuracy_score(y_true,y_pred)
    print("accuracy  %4.3f"%acc)
    print("precision %4.3f"%prec)
    print("recall    %4.3f"%recall)
    print("f1        %4.3f"%f1)
    return [prec,recall,f1,acc]
#analisa qual a melhor combinação para a analise de nossos dados, apos este chegasse a conclusao que o é melhor n=1 e weights= distance
loo = model_selection.LeaveOneOut()
# res = []
# parameters = {'n_neighbors':range(1,20),'weights': ('distance','uniform')}
# for it,(train_index,test_index)  in enumerate(loo.split(xtrain_emb)):
#     print("iteration",it)
#     clf = model_selection.GridSearchCV(model,parameters)
#     clf.fit(xtrain_emb[train_index],ytrain_array[train_index])
#     print(clf.best_params_,clf.best_score_)
#     best_model = clf.best_estimator_
#     y_pred = best_model.predict(xtrain_emb[test_index])
#     res.append(evaluate_pred(ytrain_array[test_index],y_pred))

#avalicao do modelo com parametros ajustados
model = neighbors.KNeighborsClassifier(n_neighbors = 1 , weights = "distance")
model.fit(xtrain_emb,ytrain_array)
pred = model.predict(xtest_emb)
print(f"k={i}")
print(metrics.classification_report(ytest,pred))

#criacao do modelo final e do deploy
model.fit(converter.encode(xtrain_global+xtest),ytrain_global+ytest)

from joblib import dump, load
dump(model, 'acaiteria.joblib') 
modelo_final = load('acaiteria.joblib') 

#conversor de string para valores numericos
def convert_num(n):
    valores = {'um' : 1,
               'uma' : 1,
               'dois' : 2,
               'duas' : 2,
               'tres' : 3,
               'quatro' : 4,
               'cinco' : 5,
               'seis' : 6,
               'sete' : 7,
               'oito' : 8,
               'nove' : 9,
               'dez' : 10,
               'onze' : 11,
               'doze' : 12,
               'treze' : 13,
               'quatorze' : 14,
               'catorze' : 14,
               'quinze' : 15} 
    ret = 0
    if n.isnumeric():
        ret = int(n)
    else:
        if n in valores.keys():
            ret = valores[n]

    return ret
!pip install unidecode
#utiliza do tokenizes do twitter para traduzir nossas entradas em tokens
from nltk.tokenize import TweetTokenizer
from unidecode import unidecode
tknzr = TweetTokenizer()

#processa a entrada retornando tokens que a traduzem, fiz um pre-processamento para os possiveis nomes dos produtos para facilitar o resultado
lista_entidades = [
'item:acai,acais,pequeno,pequenos,300,medio,medios,500,grande,grandes,700,acaigrande,acaipequeno,acaimedio,300ml,500ml,700ml',
'num:1,2,3,4,5,6,7,8,9,10',
'num:um,dois,tres,quatro,cinco,seis,sete,oito,nove,dez,onze,doze,quatorze,quinze,uma,duas'
]
entidades = dict()
def load_entidades(lista_entidades):
            for line in lista_entidades:
                entidade,valores = line.split(':')
                str_valores = valores[:]
                valores = str_valores.split(',')
                for valor in valores:
                    if valor not in entidades.keys():
                        entidades[valor] = entidade
load_entidades(lista_entidades)
def find_entidades(texto):
    listatokens=[]
    tokenant='auxiliar'
    ret = dict()
    for token in tknzr.tokenize(texto):
      token = token.lower()
      token = unidecode(token)
      tokenadd = token
      if tokenant == 'acai'or tokenant =='acais':
        if token =='grande' or token =='grandes' or token=='700' or token=='700ml':
          tokenadd = 'acaigrande'
        elif token =='medio' or token =='medios' or token=='500' or token=='500ml':
          tokenadd = 'acaimedio'
        elif token =='pequeno' or token =='pequenos' or token=='300' or token=='700ml' :
          tokenadd = 'acaipequeno'
      if tokenadd == 'acaipequeno' or  tokenadd == 'acaimedio'  or  tokenadd == 'acaigrande':
        listatokens.pop()
      if token in entidades.keys():
        tokenant = token;
        listatokens.append(tokenadd)
    for token in listatokens:
        if token in entidades.keys():
            ent = entidades[token]
            if ent not in ret.keys():
                ret[ent] = [token]
            else:
                ret[ent] += [token]
    return ret
#cria cardapio com valores para os possiveis nomes de produtos, o cardapio externo para o chatbot sera uma imagem importada do drive
valor_cardapio = {
    'acai':15.0,
    'acaipequeno':15.0,
    '300':15.0,
    '300ml':15.0,
    'pequeno':15.0,
    'pequenos':15.0,
    'acaimedio':20.0,
    '500':20.0,
    '500ml':20.0,
    'medio':20.0,
    'medios':20.0,
    'acaigrande':25.0,
    'grande':25.0,
    '700':25.0,
    '700ml':25.0,
    'grandes':25.0,
}
def str_menu(h):
    rstr = ''
    for item in h:
        rstr += f"{item:<10}  {h[item]:>5}\n"
    return rstr

from datetime import datetime
import time


#inicializa variaveis globais que serao necessarias para o chatbot
from datetime import datetime
pedidos = []
estado = 0
seiendereco = 0
entregando = 0

def acai(update: Update, context: CallbackContext) -> None:
    #variaveis globais reiniciadas toda vez que o pedido é encerrado
    global pedidos
    global estado
    global seiendereco
    global entregando
    global horapedido
    cliente = update.message.text
    msg = ''
    pred = modelo_final.predict([converter.encode(cliente)])[0]
    #inicia conversa de acordo com a hora do dia
    if pred == 'iniciar_conversa':
      if estado == 0:
        hora=time.strftime('%H', time.localtime())
        print(hora)
        if int(hora) > 12:
            if int(hora) > 18:
              msg +=("Boa noite, bem vindo a Açaiteria do Edu")
            else:
              msg +=("Boa tarde, bem vindo a Açaiteria do Edu")
        else:
            msg +=("Bom dia, bem vindo a Açaiteria do Edu")
      elif estado == 1:
         msg +=("Voce tem um pedido em andamento gostaria de adicionar mais alguma coisa ou pedir a conta?")
      elif estado == 2:
         msg +=("Perdão qual o endereço de entrega")
    #mostra o cardapio por foto
    elif pred == 'pedir_cardapio':
        context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
    #faz pedido e processa os tokens para parecer mais interativo
    elif pred == 'fazer_pedido':
        estado=1
        ent = find_entidades(cliente)
        if(len(ent['num'])==len(ent['item'])):
          joined = [[x,y] for x,y in zip(ent['num'],ent['item'])]
          pedidos.append(joined)
          msg += 'Certinho, então temos'
          for n,item in joined:
              msg += ', %s'%(n)
              if item == 'acaigrande' or item=='grande' or item=='grandes' or item=='700ml' or item=='700':
                msg += ' açaí grande'
              elif item == 'acaimedio'or item=='medio' or item=='medios' or item=='500ml' or item=='500':
                msg += ' açaí medio'
              elif item == 'acaipequeno'or item=='pequenos' or item=='pequeno' or item=='300ml' or item=='300':
                msg += ' açaí pequeno'
              else:
                msg += ' %s'%(item)
          msg += ' saindo\n'
        else:
          msg='Nao entendi seu pedido pode repetir por favor?'
    #processa o total e solicita endereço caso n tenha enviado ainda
    elif pred == 'fechar_conta':
      if estado==0:
        msg+=('Você ainda não pediu nada, se liga no nosso cardápio\n')
        context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
      else:
          estado=2
          pediconta=1
          msg += 'Seu pedido foi:\n'
          total = 0
          for joined in pedidos:
              for n,item in joined:
                  nvalor = convert_num(n)
                  if item in valor_cardapio.keys():
                      vitem = valor_cardapio[item]
                  total += vitem*nvalor
                  msg += '%s '%(n) 
                  if item == 'acaigrande' or item=='grande' or item=='grandes' or item=='700ml' or item=='700':
                    msg += ' açaí grande'
                  elif item == 'acaimedio'or item=='medio' or item=='medios' or item=='500ml' or item=='500':
                    msg += ' açaí medio'
                  elif item == 'acaipequeno'or item=='pequenos' or item=='pequeno' or item=='300ml' or item=='300':
                    msg += ' açaí pequeno'
                  else:
                    msg += '%s'%(item)

                  msg+=' %04.2f\n'%(vitem*nvalor)
              
          msg += ('E o valor total foi R$ %4.2f. Seu pedido esta começando a ser preparado.\n'%total)
          if seiendereco == 0:
            msg+=('Qual seu endereço?\n')
    #pega endereco e comeca a contar o tempo de preparo do pedido
    elif pred == 'passar_endereco':
      horapedido = datetime.now()
      seiendereco=1
      if estado==0:
        msg+=('Entregamos para toda cidade gratuitamente\n')
        context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
      elif estado == 1:
         msg+=('A entrega é gratuita, vamos preparar seu pedido e ja te enviamos\n')
      elif estado==2:
         msg+=('Okay, seu pedido vai ser preparado e sairá em breve\n')
    #calcula o tempo de entrega com base no tempo real, considerando o tempo de preparo/entrega de 5 minutos
    elif pred == 'tempo_entrega':
      if estado == 0:
        msg+= ('Você ainda nao fez um pedido, que tal dar uma olhada no nosso cardápio\n')
        context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
      elif seiendereco == 0:
         msg+=('Qual seu endereço?\n')
      elif seiendereco == 1:
        horaatual= datetime.now()
        tempoentrega=(horaatual-horapedido)
        tempoentrega= tempoentrega.total_seconds()
        tempoentrega=int(5-(tempoentrega/60))
        if tempoentrega>2:
          msg+=('Seu pedido ainda esta sendo preparado, a previsao para chegar em sua residencia é de ')
          msg+=str(tempoentrega)
          msg+=(' minutos.\n')
        elif tempoentrega>0:
          msg+=('Seu pedido saiu para entrega e chega em:' )
          msg+=str(tempoentrega)
          msg+=(' minutos.\n')
          entregando=1
        else:
          msg+=('O entregador ja chegou\n')
          entregando=2
    #mostra a situacao do pedido
    elif pred == 'situacao_pedido':
          if estado == 0:
            msg+= ('Você ainda nao fez um pedido, que tal dar uma olhada no nosso cardápio\n')
            context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
          elif entregando==1:
            msg+= ("Seu pedido já se encontra com o entregador, a caminho de sua casa")
          elif entregando==2:
            msg+= ("Seu pedido foi entregue")
          else:
            msg+=("Seu pedido ainda esta sendo preparado")
    else:
        msg = 'Não, entendi. \n'
        context.bot.send_document(chat_id = update.message.chat_id, document=open('/content/drive/MyDrive/trabalhoia/cardapio.jpg', 'rb'))
    #caso o pedido ja tenha sido entregue considera o fim deste e limpa todas informaçoes para guardar o proximo pedido
    if entregando==2:
      msg+=("Seu pedido foi entregue, obrigado pela preferencia")
      pedidos = []
      estado = 0
      seiendereco = 0
      entregando = 0
    update.message.reply_text(msg)

def main() -> None:
    """Start the bot."""
    updater = Updater("5168831544:AAFXvahi4WuIcxKIdp6zfoU71HZ1i99vwEw")
    # Para obter a chave converse com o BotFather no telegram (O bot cria contas de bots dentro do telegram)

    dispatcher = updater.dispatcher

   # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, acai))

    # Start the Bot
    updater.start_polling()

    updater.idle()

main()