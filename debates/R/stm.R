# STM
# install.packages("stm")
# install.packages("tm")
# install.packages("splines")
# install.packages("stmBrowser")
library('stm')
library('stmBrowser')

# ------------------------------------------------------------------------------
#  Load data
# ------------------------------------------------------------------------------

setwd("~/...")
filename <- 'texts/split_debates_1000.csv'
data     <- read.csv(filename)

# ------------------------------------------------------------------------------
#  Prepare documents
# ------------------------------------------------------------------------------

# A list of extra stopwords specific to the debates transcripts
stopwords <- c('will', 'people', 'need', 'think', 'well','going', 'can',
               'country', 'know', 'lot', 'get','make','way','president', 'want',
               'like','say','got','said','just','something','tell','put','now',
               'bad','back','want','right','every','one','use','come','never',
               'many','along','things','day','also','first','guy',
               'great', 'take', 'good', 'much','anderson')

# Stem, tokenize, stopwords, ...
processed <- textProcessor(data$text,
                           lowercase = TRUE, removestopwords = TRUE,
                           removenumbers = TRUE,
                           removepunctuation = TRUE, wordLengths = c(3,Inf),
                           striphtml = TRUE,
                           stem = TRUE,
                           metadata = data,
                           customstopwords = stopwords)

# Remove words that are too frequent if needed
# plotRemoved(processed$documents, lower.thresh=seq(from = 10, to = 1000, by = 10))

# Prepare documents for analysis
out   <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs  <- out$documents
vocab <- out$vocab
meta  <- out$meta
meta$party <- as.factor(meta$party)
meta$candidate <- as.factor(meta$candidate)

# ------------------------------------------------------------------------------
#  Find optimum number of topics
# ------------------------------------------------------------------------------

# Grid search
n_topics = seq(from = 5, to = 12, by = 1)

storage <- searchK(out$documents, out$vocab, K = n_topics,
                   prevalence  =~ party, data = meta)
plot(storage)
print(storage)

# Select the best number of topics that maximizes both exclusivity
# and semantic coherence
plot(storage$results$exclus, storage$results$semcoh)
text(storage$results$exclus, storage$results$semcoh, labels=storage$results$K,
     cex= 0.7, pos = 2)

# Set # topics
n_topics = 8

# ------------------------------------------------------------------------------
#  Model selection
# ------------------------------------------------------------------------------
debateSelect <- selectModel(out$documents, out$vocab,
                              K           = n_topics,
                              prevalence  =~ candidate,
                              data        = meta,
                              runs        = 20,
                              seed        = 1)

debateSelect
plotModels(debateSelect)

# Choosing the second model as maximizing Exclusivity and Semantic Coherence
debateFit  <- debateSelect$runout[[2]]

# ------------------------------------------------------------------------------
#  Model exploration and validation
# ------------------------------------------------------------------------------

plot(debateFit, labeltype=c("frex"))
topicQuality(model=debateFit, documents=docs)

# List of words associated with the topic
sageLabels(debateFit, n=10)

labelTopics(debateFit, n=8)

# ------------------------------------------------------------------------------
#  Model exploration and validation: figure
# ------------------------------------------------------------------------------

plot(storage)

attach(mtcars)
par(mfrow=c(2,2))

# 1) Choosing the number of topics: Exclusivity vs Semantic Coherence
plot(storage$results$semcoh,storage$results$exclus, ylab='Exclusivity', xlab='Semantic Coherence', main='Number of topics',bty="n")
grid(lwd = 1)
text( storage$results$semcoh, storage$results$exclus,labels=storage$results$K,cex= 0.7, pos = 2,bty="n")
# 2) Model Selection
plotModels(debateSelect, main='Model Selection - Best Likelihood',bty="n")
grid(lwd = 1)
# 3) Validating Topic Quality
topicQuality(model=debateFit, documents=docs, main='Topic Quality',bty="n")
grid(lwd = 1)
plot(debateFit, labeltype=c("frex"), main = 'Topic Most Frequent Words',bty="n")

grid(lwd = 1)


# We find the following topics
topics = c('Elections', 'Climate change', 'Planned Parenthood','Middle East',
           'Taxes / Drugs','Business (Trump)','Social Security','Immigration')

# ------------------------------------------------------------------------------
#  Influence of Party on topic prevalence
# ------------------------------------------------------------------------------
prep <- estimateEffect(1:n_topics ~ party, debateFit, meta=meta,
                       uncertainty = "Global")

plot.estimateEffect(prep, covariate = "party", model=debateFit,
                    method="pointestimate",
                    topics = c(8, 3,2,4),
                    xlim=c(-0.1,0.40),
                    xlab="",
                    main="Effect of Liberal vs. Conservative",
                    labeltype = "frex",
                    bty="n"
)

# ------------------------------------------------------------------------------
#  Influence of candidate + party covariate on topic prevalence
#  D3 visualization
# ------------------------------------------------------------------------------

debateSelect <- selectModel(out$documents, out$vocab,
                            K           = n_topics,
                            prevalence  =~ party + candidate,
                            data        = meta,
                            runs        = 20,
                            seed        = 1)

debateSelect
plotModels(debateSelect)

debateFit_candidate  <- debateSelect$runout[[2]]

debateFit_candidate <- stm(out$documents, out$vocab,
                            K           = n_topics,
                            init.type   = 'Spectral',
                            prevalence  =~ candidate,
                            data        = meta,
                            seed        = 1)


# In browser
stmBrowser(debateFit_candidate, data=out$meta,
            c("candidate","party"), text="text", labeltype='frex')



