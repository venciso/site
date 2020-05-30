---
title: Syntax Highlighting in Blogdown; a very specific solution
author: ''
date: '2020-05-29'
slug: syntax-highlighting-blogdown
categories: []
tags:
  - R
  - Blogdown
  - Hugo
  - Syntax highlighting
publishdate: '2020-05-29'
lastmod: '2020-05-29'
---

If you spend more than 5 seconds on this site you will be able to tell that it is not one of the snazziest ones around. This is mostly by design but also a because I know very little about web development. 

These days it is *really* easy to have your own R website thanks to `blogdown`. `blogdown` interfaces with `Hugo` to let you have a working site up and running in minutes. A good tutorial to get started can be found [here](http://jonthegeek.com/2018/02/27/blogging-in-ten-minutes/).

When I decided to build this site I knew I wanted a simple design and that I didn't want to mess about too long with setting it up and so I went looking for `Hugo` [themes](https://themes.gohugo.io/) and I settled on [this one](https://themes.gohugo.io/minimal-bootstrap-hugo-theme/). 

As you can see I've only got three pages; posts, tags and about. I'd rather like to add an archive and maybe a search bar but the point is I'm happy with the basic structure I've got. What's important to me is that the posts render properly and that they are readable. 

Which is why I wanted to add syntax highlighting to my posts. Without it the code chunks in your post look like this:

```
xgboostParams <- dials::parameters(
  min_n(),
  tree_depth(),
  learn_rate(),
  finalize(mtry(),select(proc_mwTrainSet,-outcome)),
  sample_size = sample_prop(c(0.4, 0.9))
)
```

It is functional but it makes the post look a bit samey. You can play around with the colour of the text to help differentiate between code and not-code.

If you apply syntax highlighting you end up with something more like this: 


```r
xgboostParams <- dials::parameters(
  min_n(),
  tree_depth(),
  learn_rate(),
  finalize(mtry(),select(proc_mwTrainSet,-outcome)),
  sample_size = sample_prop(c(0.4, 0.9))
)
```

This looks much nicer in my opinion and makes the post more readable. 

So, how do you do it? 

The answer won't be universal but if you are lucky and the theme you're using already supports it then this might save you some googling.

## TL;DR 

When creating the a new post through the `blogdown` Addins be sure to select **Rmarkdown** as a format and not **Rmd**.

![](/post/2020-05-29-syntax-highlighting-in-blogdown-a-very-specific-solution_files/select_rmarkdown.gif)

### A bit more detail

To anyone with some knowledge of `Hugo` the above will be completely obvious and even silly but actually it took me longer than I'd care to admit to get to the answer.

First, I knew that it should be possible to have syntax highlighting in my theme because it is mentioned on the theme's [page](https://themes.gohugo.io/minimal-bootstrap-hugo-theme/):

>Hugo has built-in syntax highlighting, provided by Chroma. It is currently enabled in the config.toml file from the exampleSite.
Checkout the Chroma style gallery and choose the style you like.

Also, the `config.toml` file contains this section which is the bit that actually parametrises the highlighting. 

```
 [markup.highlight]
    codeFences = true
    hl_Lines = ""
    lineNoStart = 1
    lineNos = false
    lineNumbersInTable = true
    noClasses = true
    style = "solarized-dark"
    tabWidth = 4
```

In the code above `solarized-dark` is the name of the `Chroma` highlighting style. All the available styles can be found [here](https://xyproto.github.io/splash/docs/all.html). 

However, I didn't know how to activate it. In fact according to that description it should come activated by default but none of the posts I had created displayed any highlighting. 

After some more googling I stumbled onto [this section](https://bookdown.org/yihui/blogdown/output-format.html) of the **Creating Websites with R Markdown** book which outlines the differences between the `Rmd` and `Rmarkdown`formats. 

Turns out that each format is rendered to HTML through different converters. `Rmarkdown` uses something called `Blackfriday` and `Rmd` uses `Pandoc`. As I understanding then `Rmd` is rendered by `R` and `Rmarkdown` is rendered by `Hugo` and so posts need to be rendered by `Hugo` in order for all the configs in the .toml file to apply. 

In the aforementioned book the authors call out some limitations with `Rmarkdown`; namely that it does not support bibliography nor does it support HTML widgets. 

The second one of those is more relevant to my site as I have at least one post that uses widgets. For example, [this post](https://venciso.netlify.app/2020/05/virtual-madrid-open/) contains a `leaflet` map which is not rendered if I use `Rmarkdown`. This means that for now if I want to use HTML widgets I'll have to sacrifice syntax highlighting in those posts. Having said that, I am sure that somebody knows how to apply highlighting to `Rmd` files but for now I'm ok with the compromise. 

One more thing I should say is that my site's theme requires Hugo version `0.60.1` as a minimum which is quite a recent one. In older posts I found on this issue such as [this one](https://discourse.gohugo.io/t/cant-get-syntax-highlighting-to-work/15350) there are references to parameters like `pygmentsCodefences` and `pygmentsStyle` so if your theme is running on an older Hugo version this might be of help.

Also, if your site's theme doesn't already come with syntax highlighting [this post](https://amber.rbind.io/2017/11/15/syntaxhighlighting/) might help you out. It goes into quite a bit of detail on how to add `highlight.js`.

That's all I've got for now. I hope this is useful to at least one other `R` user lost in the in and outs of how `Hugo` works. 
