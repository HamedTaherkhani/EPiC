thislist <- c(93,93,93,93,93,94,95,95,95,95)
hist(thislist)
wilcox.test(thislist-6,
alternative = "greater",
  mu = 87 # default value
)
wilcox.test(thislist-1,
alternative = "greater",
  mu = 92 # default value
)
wilcox.test(thislist-2,
alternative = "greater",
  mu = 91 # default value
)