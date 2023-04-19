---
title: 'Slides'
defaultTemplate: "[[anl-template]]"
center: true
previewLinks: true
width: 960
height: 700
margin: 0.1
css:
 - "./custom.css"
---

<!-- .slide template="[[anl-template]]" bg="#1c1c1c" style="align:center; justify:center;" -->


<grid
    align="center"
    drop="center"
    drag="70 20"
	style="background-color:#282828; border-radius:8px; box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;">

## <span style="font-size:1.75em;">Generative Modeling and Efficient Sampling Techniques </span><!-- .element style="font-weight:900; line-height:0.6em;" -->

<a href="https://github.com/saforem2/Notes-Demo/blob/master/slides/demo/slides.md">
<i class="fab fa-github fa-1x" ></i>
<code><span style="color:#bdbdbd!important; background-color: #282828;">saforem2/deep-fridays</span></code>
</a>

</grid>

<grid drag="100 10" drop="0 65" align="bottomleft" >

<a href="https://samforeman.me">

<i class="fas fa-home" style=""></i>
<span style="color: #BDBDBD;">Sam Foreman</span>
</a>

<span style="font-size:0.8em; color:#505050; padding:0px; margin:0px; text-align:center!important;">2023-04-21</span>

</grid>

<grid drag="100 10" drop="bottomright" align="bottomright" >
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/Argonne_cmyk_white.svg" alt="Argonne National Laboratory">
</grid>
<grid drag="100 10" drop="bottom" align="bottomright" >
<!--<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/Argonne_cmyk_white.svg" alt="Argonne National Laboratory">-->
</grid>

---
<!-- .slide bg="#1c1c1c" style="text-align:left; -->

<grid drag="100 100" drop="0 0" align="topleft" style="">

<grid drag="60 100" drop="0 0" style="text-align:left; font-size:1.2em;">

# Overview
- Standard Model:
	-  ⚡🧲 E \& M 
	- ⚛️ Nuclear interactions
		- strong / weak force
- Quantum Chromodynamics (**QCD**):
	- Interaction between quarks and gluons
	- Analytical progress is _difficult_...

</grid>

<grid drag="40 60" drop="60 25" style="">
![nucleus](https://github.com/saforem2/physicsSeminar/raw/main/assets/static/nucleus.svg) <!-- .element style="width:25%;" -->
![feynman](https://github.com/saforem2/physicsSeminar/raw/main/assets/static/feynman.svg) <!-- .element style="width:80%;" -->
</grid>

</grid>


---

<!-- .slide bg="#1c1c1c" style="text-align:left; -->

<grid drag="90 75" drop="10 10" align="topleft" style="font-size:1.1em;">

# Overview

- Standard Model:
	-  E\&M ⚡🧲 
	- ⚛️ Nuclear interactions (strong / weak force)
- _Quantum Chromodynamics_ (**QCD**):
	- Strong interaction between quarks and gluons in the nucleus
	- Analytical progress is _difficult_...


<grid drag="100 45" drop="bottom" style="width:flex;" align="center">

<split left="1" right="3">

![nucleus](https://github.com/saforem2/physicsSeminar/raw/main/assets/static/nucleus.svg) <!-- .element style="width:25%;" -->
![feynman](https://github.com/saforem2/physicsSeminar/raw/main/assets/static/feynman.svg)
</split>
</grid>

</grid>

---
<!-- .slide bg="#1c1c1c" style="text-align:center; font-size:1.1em;" -->

# Markov Chain Monte Carlo (MCMC)

<p><b>Goal</b>: Generate <b>independent</b> samples distributed according to $\{x_{i}\} \sim p(x)$ </p></span> <!-- .element style="display: table; margin: var(--r-block-margin) 0!important; background-color:#66bb6a; color:#000; border-radius:4px; padding:4px 4px; line-height:1.25em; margin-left: auto; margin-right: auto; text-align:left!important; max-width: fit-content;" -->

- $p(x)$ is the **target distribution**[^1]:
   
  \begin{equation}
  p(x) \propto e^{-S(x)}
  \end{equation}

- Want to calculate observables $\mathcal{O}$

  \begin{equation}
  \left\langle \mathcal{O}\right\rangle \propto \int \left[\mathcal{D}x\right]\hspace{4pt} {\mathcal{O}(x) e^{-S[x]}}
  \end{equation}

- If we had <span style="color:#00CCFF;">independent</span> configurations, we could approximate the integral, with error $\sigma_{\mathcal{O}}^{2}$

  \begin{equation}
  \hspace{20pt} \left\langle\mathcal{O}\right\rangle \textcolor{#00CCFF}{\simeq} \frac{1}{N}\sum_{n=1}^{N} \mathcal{O}(x_{n})
  \quad \textcolor{#00CCFF}{\Longrightarrow}\quad \sigma^{2}_{\mathcal{O}} = \frac{1}{N} \mathrm{Var}\left[\mathcal{O}(x)\right]
  \end{equation}

[^1]: Here, $S(x)$ is the action (~ potential energy) of our physical system

</grid>

<grid drag="40 30" drop="65 30" style="">

![](file:///Users/samforeman/Obsidian/Notes/Excalidraw/normal_distribution.dark.png)
</grid>


---

<grid drag="100 100" style="font-size:0.9em; line-height: 0.9em;">

# Metropolis-Hastings <!-- .element style="margin-left:15%; margin-bottom:1em; font-size: 2.0em;" -->

:::

```python
import numpy as np

def prob(x: float) -> float:
    denom = np.sqrt(2 * np.pi ** 2)
    return np.exp(-0.5 * (x ** 2)) / denom

def metropolis_hastings(nsteps: int = 1000):
    x = 0.  # initialize config
    samples = []
    for n in range(nsteps):
        xp = x + np.random.randn()    # generate random proposal
        likelihood_ratio = prob(xp) / prob(x)
        # always accept if likelihood_ratio > 1
        # otherwise, accept according to ⤵️ (for stochasticity)
        if np.random.rand() < likelihood_ratio:
            x = xp
        samples.append(x)
    return samples
```

::: <!-- .element style="font-size:0.6em; line-height: 0.8em!important;" -->


<grid drag="20 20" drop="64 18" style="border:1px solid #66BB6A; zindex:1; border-radius:8px; box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;">

\begin{equation}
p(x) = \frac{e^{-\frac{1}{2}x^{2}}}{\sqrt{2\pi}}
\end{equation}

</grid>


&nbsp;
											   
```python
>>> %timeit metropolis_hastings(int(1e6))  # 2023 MBP M2 Max
1.92 s ± 9.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

<!--$1\times 10^{6}$ samples in $\sim 2$ s !!-->


</grid>

---

# Results

<span id="large">

As $N\rightarrow\infty$, $\quad x \sim\mathcal{N}(0, \mathbb{1})$

</span>

<div align="center">
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-10.svg" width=32% align="center">
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-100.svg" width=32% align="center">
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-500.svg" width=32% align="center">
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-1000.svg" width=32% align="center">
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-10000.svg" width=32% align="center">
<!-- <img src="assets/static/normal/samples-1e5.svg" width=32% align="center"> -->
<img src="https://raw.githubusercontent.com/saforem2/physicsSeminar/main/assets/static/normal/samples-1e+06.svg" width=32% align="center">
</div>

</div>

---

# Hamiltonian Monte Carlo (HMC)<!-- .element style="margin-left:auto; margin-right:auto;" --> 


- Want to (sequentially) construct a chain of states $\\{x_{i}\\}$,  
  such that  as $N \rightarrow \infty$:
 
  \begin{equation}
  \\{x_{0}, x_{1}, x_{2}, \cdots \\} \rightarrow p(x)
  \end{equation}

- <span style="color:#63ff5b;">Trick:</span>

	- Recall, $p(x) = {e^{-S(x)}} / {Z}$
	- Introduce _fictitious_ momentum $v \sim \mathcal{N}(0, \mathbb{1})$
	- Normally distributed **independent** of $x$, i.e.
 
  \begin{align}
  	p(x)p(v) &\propto e^{-S{(x)}} e^{-\frac{1}{2} v^{T}v}\\\\
   	&= e^{-\left[S(x) + \frac{1}{2} v^{T}{v}\right]} \\\\
   	&\textcolor{#63ff5b}{=} e^{-H(x, v)}
  \end{align}

---

<!-- .slide style="text-align:left;" -->

# Hamiltonian Monte Carlo (HMC)

- $p(x, v) = p(x) p(v) \propto e^{-H(x, v)}$, where

    - $p(x, v)$ is the joint distribution[^1], and

    - $H(x, v)$ is the **Hamiltonian**
 
- We can evolve the $(\dot{x}, \dot{v})$ system to get new states $\\{x_{i}\\}$ !!

- Hamiltonian Dynamics:

\begin{equation}
  \dot{x} = \frac{\partial H}{\partial v}
  \quad \quad
  \dot{v} = - \frac{\partial H}{\partial x}
\end{equation}


![](./assets/hmc1.svg)  <!-- .element width="80%" -->

[^1]: Note that we simply discard the momentum at the end of the trajectory.

---
<section data-background-iframe="https://chi-feng.github.io/mcmc-demo/app.html"
          data-background-interactive>
</section>

---
<section data-background-iframe="file:///Users/samforeman/projects/mcmc-demo/app.html" data-background-interactive style="border:none; width:100%; color:#1c1c1c!important; background-color:#1c1c1c;">
</section>

---

<!-- .slide bg="#1c1c1c" -->
# Callouts

- Icons are broken for some reason???

> <i class="fas fa-bug"></i> **Bug**<br>
> Here is an Example for an Callout in a Slide. Callouts support dark and white backgrounds and could be sized by annotations

---

<!-- .slide bg="#1c1c1c" style="text-align:left!important; align:left!important" -->

# Thank you!

- Feel free to reach out! <br>
<split gap="2">

   [<i class="fab fa-twitter"></i>](https://www.twitter.com/saforem2)
   [<i class="far fa-paper-plane"></i>](mailto:///foremans@anl.gov)
   [<i class="fas fa-home"></i>](https://samforeman.me)
</split>

> [!INFO] Acknowledgements
> This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
<!-- .element style="max-width:90%; text-align:left;" -->

---

<style>

:root {
    --callout-radius:5px;
    --admonition-margin-top: 1.5625em;
    --admonition-margin-bottom: var(--admonition-margin-top);
    --admonition-margin-top-lp: 0px;
    --admonition-margin-bottom-lp: 12px;
    --r-math-color:#FAFAFA;
    --cm-inline-background: #242424;
    --cm-inline-foreground: #00CCFF;
    --r-heading-text-transform: none;
    --primaryBorderColor: #666666;
    --r-main-background-color: #1c1c1c!important;
    --r-heading-letter-spacing: -0.45px;
    --r-heading-word-spacing: 0.5px;
    --r-heading-text-transform: none;
    --r-heading-text-shadow: none;
    --r-heading-font-weight: 700;
    --r-heading1-text-shadow: none;
    --r-main-font-size: 22px;
    --r-main-line-height: 1.5em;
    --r-monospace-font-size: 18px;
    --r-heading1-size: 1.33em;
    --r-heading2-size: 1.25em;
    --r-heading3-size: 1.2em;
    --r-heading4-size: 1.15em;
    --r-heading5-size: 1.05em;
    --r-heading6-size: 1.025em;
    --r-heading-line-height:1.5em;
    --r-main-font-family: 'Inter';
    --r-code-font: "JuliaMono", "agave Nerd Font", "Monaco", "Hack", "VictorMono", monospace;
    --r-link-color: #03A9F4;
    --r-link-color-dark: #f92672;
    --r-link-color-hover: #63ff51;
    --r-accent-color: #77CA29;
    --r-controls-color: #228BE6;
    --r-progress-color: #404040;
    --r-header-accent: #00CCFF;
    --r-selection-background-color: RGBA(255, 255, 0, 0.15);
    --r-selection-color: RGB(255, 255, 0);
    --r-main-color: #c8c8c8;
    --text-muted: #757575;
    --text-faint: #404040;
    --r-heading-color: #FFF;
    --r-background-color: #1c1c1c;
    --cm-keyword: #c792ea;
    --cm-atom: #f78c6c;
    --cm-number: #ff5370;
    --cm-type: #decb6b;
    --cm-def: #82aaff;
    --cm-property: #c792ea;
    --cm-variable: #f07178;
    --cm-variable-2: #EEFFFF;
    --cm-variable-3: #f07178;
    --cm-definition: #82aaff;
    --cm-callee: #89ddff;
    --cm-qualifier: #decb6b;
    --cm-operator: #89ddff;
    --cm-hr: #98e342;
    --cm-link: #696d70;
    --cm-header: #da7dae;
    --cm-builtin: #ffcb6b;
    --cm-meta: #ffcb6b;
    --cm-matching-bracket: #FFFFFF;
    --cm-tag: #ff5370;
    --cm-tag-in-comment: #ff5370;
    --cm-string-2: #f07178;
    --cm-bracket: #ff5370;
    --cm-comment: #676e95;
    --cm-string: #c3e88d;
    --cm-attribute: #c792ea;
    --cm-attribute-in-comment: #c792ea;
    --cm-background-color: #1c1c1c;
    --cm-active-line-background-color: #353a50;
    --cm-foreground-color: #AE81FF;
    --code-normal: #AE81FF;
    -webkit-font-smoothing:subpixel-antialiased;
    --font-smoothing:subpixel-antialiased;
    --chart-color-1: #ff00ff;
    --chart-color-x: RGB(255.0,255.0,255.0);
	  --admonition-margin-top: 1.5625em;
	  --admonition-margin-bottom: var(--admonition-margin-top);
	  --admonition-margin-top-lp: 0px;
	  --admonition-margin-bottom-lp: 12px;
	  --callout-border-width: 0px;
	--callout-border-opacity: 0.25;
	--callout-padding: var(--size-4-3) var(--size-4-3) var(--size-4-3) var(--size-4-6);
	--callout-radius: var(--radius-s);
	--callout-blend-mode: var(--highlight-mix-blend-mode);
	--callout-title-color: inherit;
	--callout-title-padding: 0;
	--callout-title-size: inherit;
	--callout-content-padding: 0;
	--callout-content-background: transparent;
	--callout-bug: var(--color-red-rgb);
	--callout-default: var(--color-blue-rgb);
	--callout-error: var(--color-red-rgb);
	--callout-example: var(--color-purple-rgb);
	--callout-fail: var(--color-red-rgb);
	--callout-important: var(--color-cyan-rgb);
	--callout-info: var(--color-blue-rgb);
	--callout-question: var(--color-yellow-rgb);
	--callout-success: var(--color-green-rgb);
	--callout-summary: var(--color-cyan-rgb);
	--callout-tip: var(--color-cyan-rgb);
	--callout-todo: var(--color-blue-rgb);
	--callout-warning: var(--color-orange-rgb);
	--callout-quote: 158, 158, 158;
	--callout-border-opacity: 0.25;
	--callout-padding: var(--size-4-3) var(--size-4-3) var(--size-4-3) var(--size-4-6);
	--callout-radius: var(--radius-s);
	--callout-blend-mode: var(--highlight-mix-blend-mode);
	--callout-title-color: inherit;
	--callout-title-padding: 0;
	--callout-title-size: inherit;
	--callout-content-padding: 0;
	--callout-content-background: transparent;
	--callout-bug: var(--color-red-rgb);
	--callout-default: var(--color-blue-rgb);
	--callout-error: var(--color-red-rgb);
	--callout-example: var(--color-purple-rgb);
	--callout-fail: var(--color-red-rgb);
	--callout-important: var(--color-cyan-rgb);
	--callout-info: var(--color-blue-rgb);
	--callout-question: var(--color-yellow-rgb);
	--callout-success: var(--color-green-rgb);
	--callout-summary: var(--color-cyan-rgb);
	--callout-tip: var(--color-cyan-rgb);
	--callout-todo: var(--color-blue-rgb);
	--callout-warning: var(--color-orange-rgb);
	--callout-quote: 158, 158, 158;
}

.standout{
    background: var(--cm-background-color);
    padding:5px;
    font-weight:700;
    border-radius:6px;
}

.reveal pre {
  display:block;
  margin:auto;
  width:auto;
  font-family: var(--r-code-font);
  font-size: var(--r-monospace-font-size);
  padding: auto;
  white-space: pre-wrap;
}

.reveal p {
  margin:auto!important;
  padding:auto!important;
}

.reveal pre code {
    display: inline-block;
    top: 2px;
    white-space: pre;
    bottom: 2px;
    margin:auto;
    padding:auto;
    font-size: 0.8em;
    background:var(--cm-background-color);
    color: var(--cm-foreground-color)!important;
    text-align: justify;
    letter-spacing: -0.45px!important;
    word-spacing: -0.5px!important;
}

.reveal {
    font-family: var(--r-main-font), sans-serif;
    font-size: var(--r-main-font-size);
    font-weight: normal;
    color: var(--r-main-color);
    background-color: var(--r-main-background-color);
}

.reveal blockquote p {
    color: var(--text-muted);
	box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15x;
    font-style: normal !important;
    font-align: left;
    display: inline;
    text-align: left;
}

.reveal blockquote em{
  color: var(--text-muted);
  text-align: left;
}

.reveal blockquote {
  border-radius: 8px !important;
  margin: 0.5rem 0rem 0.5rem 0rem;
  text-align: left;
  padding-top: 1rem;
  padding-left: 2rem;
  padding-bottom: 1rem;
  padding-right: 2rem;
  width: auto;
  font-style: normal !important;
}

.reveal blockquote {
	font-size: unset;
	margin: auto;
	padding:auto;
}


.reveal ul, ol {
    text-align:left;
}

.reveal ul ul,
.reveal ul ol,
.reveal ol ol,
.reveal ol ul {
  text-align:left;
}

.reveal ul ul {
    list-style: circle;
}
.container {
  position: relative;
}

.make-it-pop {
  filter: drop-shadow(0 0 10px purple);
}

@media (max-width: 95%) {
  section {
    -webkit-flex-direction: column;
    flex-direction: column;
  }
}

.footer {
  font-size: 60%;
  vertical-align:bottom;
  color:#bdbdbd;
  font-weight:400;
}
.note {
  color:var(--callout-color);
  border-radius:8px;
  background-color:#35353540;
  width: max-content;
  border-color:#66666640;
  padding: auto;
  margin:auto;
}

#blue {
  color: #00CCFF;
}

#red {
  color: #FF5252;
}

.callout {
  border-style: none;
  border-color: RGBA(var(--callout-color), var(--callout-border-opacity));
  border-width: var(--callout-border-width);
  border-radius: var(--callout-radius);
  margin: 1em 0;
  mix-blend-mode: var(--callout-blend-mode);
  background-color: RGBA(var(--callout-color), 0.1);
  padding: var(--callout-padding);
}
.callout-title {
  font-size: var(--callout-title-size);
  color: RGB(var(--callout-color));
  background-color: RGB(var(--callout-color), 0.0);
  line-height: var(--line-height-tight);
  font-weight: 700;
}
.callout-content {
  overflow-x: auto;
  padding: auto;
  background-color: var(--callout-content-background);
}
.callout-icon {
  flex: 0 0 auto;
  padding: auto;
  display: flex;
  align-self: center;
  content:
}
.callout-icon .svg-icon {
  color: RGB(var(--callout-color));
}

.callout-title-inner {
  font-weight: var(--bold-weight);
  color: var(--callout-title-color);
}
.callout-fold {
  display: flex;
  align-items: center;
  padding-right: var(--size-4-2);
}

.reveal .code-wrapper code {
	width: 98%;
}
.reveal code {
  font-family: var(--r-code-font);
  text-transform: none;
  tab-size: 4;
  background-color:var(--cm-background-color);
  color: var(--cm-foreground-color);
  border-radius:2px;
  letter-spacing: -0.45px!important;
  word-spacing: -0.5px!important;
}

.reveal pre code {
	padding: auto;
	border:none;
	border-radius:2px;
	font-size:0.9em;
	margin:auto;
	background: var(--cm-background-color);
}
.reveal p code {
  font-family: var(--r-code-font);
  text-transform: none;
  tab-size: 4;
  padding:auto;
  font-size:0.9em!important;
  line-height:inherit;
  background:var(--cm-inline-background);
  color: var(--cm-inline-foreground);
  border-radius:3px;
  letter-spacing: -0.45px!important;
  word-spacing: -0.5px!important;
}

mjx-container[jax="CHTML"][display="true"] mjx-math {
  /*color: var(--r-math-color);*/
  color:inherit!important;
  font-size:1.1em;
}

mjx-math {
  color: inherit!important;
  background: none!important;
  padding:unset;
  vertical-align:inherit;
}
#customcontrols > ul {
  display: none!important;
}

#customcontrols button {
  display: none!important;
}

.reveal .slides > section.present, .reveal .slides > section > section.present {
  min-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  /*justify-content: center !important;*/
  position: absolute !important;
  top: 0 !important;
}

section > h1 {
  position: absolute !important;
  top: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
  left: 0 !important;
  right: 0 !important;
}

h1 {
	border-bottom: 2px solid var(--r-header-accent);
	/*text-align: left!important;*/
	min-width: max-content;
	max-width: min-content;
}

.print-pdf .reveal .slides > section.present, .print-pdf .reveal .slides > section > section.present {
  min-height: 770px !important;
  position: relative !important;
}

.hljs {
    background: var(--cm-background-color) !important;
	font-size:inherit;
}

.hljs-float { 
	color: #00BFA5;
}

.hljs-main {
	color: #ff5252
}

.hljs-built_in {
  color: #63ff5b;
}

.hljs-comment, .hljs-quote, .hljs-deletion {
	color: #454545;
}

.hljs-params{
	color: #03A9F4;
}

.hljs-meta {
	color: #AE81FF;
}

.hljs-string {
	color: #FFFF00;
}

strong {
	color: #FF5252!important;
	font-weight: 700;
}

ul {
	margin-left: 0;
	padding-left:1em;
}

ol {
	margin-left: 0;
	padding-left:1em;
}

.reveal .code-wrapper code {
 st white-space: unset;
}

.reveal pre {
white-space: pre-wrap;
}


.reveal sup {
	font-size:0.6em;
}

.markdown-rendered code {
  color: var(--code-normal)!important;
  font-family: var(--font-monospace);
}

body.fallback-highlighting[class*="theme-"] .markdown-preview-view pre.cm-s-obsidian[class*="language-"], body.fallback-highlighting[class*="theme-"] .markdown-preview-view code[class*="language-"], body.fallback-highlighting[class*="theme-"] .markdown-preview-view .HyperMD-codeblock, body.fallback-highlighting[class*="theme-"] .markdown-preview-view .cm-hmd-codeblock {
  --font-monospace: var(--cm-font-monospace);
  color: var(--cm-foreground-color);
  font-family: var(--cm-font-monospace);
  font-weight: var(--cm-font-weight);
  line-height: var(--cm-line-height);
  font-size: var(--cm-font-size);
  white-space: var(--cm-wrap-lines);
}

.reveal .code-wrapper code {
  color: #B0BEC5;
  font-family: var(--r-code-font);
  font-size: 18px;
  line-height:1.5em;
}
/* Content */

/* Blockquotes */

.markdown-preview-view blockquote {
	padding: 0 0 0 var(--nested-padding);
	font-size: var(--blockquote-size);
}
.markdown-source-view.mod-cm6.is-live-preview .HyperMD-quote,
.markdown-source-view.mod-cm6 .HyperMD-quote {
	font-size: var(--blockquote-size);
}
.is-live-preview .cm-hmd-indent-in-quote {
	color: var(--text-faint);
}

/* Callouts */

.is-live-preview.is-readable-line-width > .cm-callout .callout {
	max-width: var(--max-width);
	margin: 0 auto;
}

.callout-icon {
	display: none;
}

.callout {
	background-color: var(--background-primary);
	margin-top: -24px;
	z-index: 200;
	width: fit-content;
	padding: 0 0.5em;
	margin-left: -0.75em;
	letter-spacing: 0.05em;
	font-variant-caps: all-small-caps;
}

.callouts-outlined .callout .callout-title {
	background-color: var(--background-primary);
	margin-top: -24px;
	z-index: 200;
	width: fit-content;
	padding: 0 0.5em;
	margin-left: -0.75em;
	letter-spacing: 0.05em;
	font-variant-caps: all-small-caps;
}
.callouts-outlined .callout {
	overflow: visible;
	--callout-border-width: 1px;
	--callout-border-opacity: 0.5;
	--callout-title-size: 0.8em;
	--callout-blend-mode: normal;
	background-color: transparent;
}

.callouts-outlined .cm-embed-block.cm-callout {
	padding-top: 12px;
}

.callouts-outlined .callout-content .callout {
	margin-top: 18px;
}

.callout[data-callout="custom-question-type"] {
	--callout-color:  '#FF5252';
	 --callout-icon: lucide-alert-circle;
 }

.callout[data-callout="my-comment"] { --callout-color: 99, 71, 214; --callout-icon: lucide-message-square; /* --callout-icon: message-o; // font-awesome */ border: 3px solid rgba(0, 0, 0, 0.5); border-radius: 5px; box-shadow: 20px 20px 40px rgba(255, 0, 0, .5); }

.row {
  display: flex;
}

.column {
  flex: 50%;
}


#left {
  margin: 0 0 5px 5px;
  text-align: left;
  float: left;
  z-index: -10;
  width: 48%;
  font-size: 0.85em;
}

#right {
  margin: 0 0 5px 0;
  float: right;
  max-width: 48%;
  text-align: left;
  z-index: -10;
  width: 48%;
  font-size: 0.85em;
}

#darkBack {
    background-color: #1c1c1c;
    color: #efefef;
    .reveal a {
        color: #F92672;
        transition: color 0.15s ease;
    }
    .reveal a:hover {
        color: var(--r-link-color-hover);
    }
}

.multiCol {
    display: table;
    table-layout: fixed; /* don't fudge depending on content */
    width: 100%;
    text-align: left; /* matter of taste, makes imho sense */
    .col {
        display: table-cell;
        vertical-align: top;
        width: 50%;
        padding: 2% 0 2% 3%; /* some vertical, and between columns */
        &:first-of-type { padding-left: 0; } /* there's nothing before col1 */
    }
}

.footnotes {
	font-size:0.8em;
	border-top: 1px dashed #454545;
	text-align: justify;
	min-width: max-content;
	padding-top: 1em;
	margin-left:0px;
	padding-left: 0px;
	padding-right: 15px;
}

.reveal sup {
	color: var(--r-link-color);
	vertical-align: top;
	position: relative;
	font-weight:800;
	font-size:0.8em;
}

.reveal em {
	color: #2196F3;
	font-weight: 700;
}

.footnotes > ol > li::marker {
    color: var(--r-link-color);
    content: counter(list-item) ": ";
    text-align: left;
}

.reveal .code-wrapper {
	width: fit-content;
	display: inherit;
	white-space: normal;
}

:root {
  --hljs-selection: #ffff0020;
  --hljs-background: #1c1c1c;
  --hljs-text: #BDBDBD;
  --hljs-string: #ffff00;
  --hljs-number: #0dcdcd;
  --hljs-title: #40c057;
  --hljs-built_in: #F39CF4;
  --hljs-keyword: #f92672;
  --hljs-function: #FF79FF;
  --hljs-params: #97F5D3;
  --hljs-comment: #404040;
}

/* Dark Orange Theme */

pre code {
  /*display: block;*/
  overflow-x: auto;
  line-height: inherit;
  font-size:inherit;
  background: var(--background);
  /*-webkit-text-size-adjust: none;*/
}

pre code *::selection,
.hljs::selection {
  background: var(--hljs-selection) !important;
}

.hljs {
  color: var(--hljs-text);
}

.hljs-string {
  color: var(--hljs-string);
}

.hljs-number {
  color: var(--hljs-number) !important;
}

.hljs-title {
  color: var(--hljs-title) !important;
}

.hljs-built_in {
  color: var(--hljs-built_in) !important;
}

.hljs-keyword {
  color: var(--hljs-keyword) !important;
}

.hljs-function > .hljs-keyword {
  color: var(--hljs-function) !important;
  font-style: italic;
}

.hljs-params,
.hljs-params,
.hljs-title.class_,
.hljs-class .hljs-title{
  color: var(--hljs-params);
}

.hljs-comment,
.hljs-deletion,
.hljs-meta {
  color: var(--hljs-comment);
}

.reveal ol,
.reveal dl,
.reveal ul {
	display: inline-block;
	text-align: left;
	margin: 0 0 0 1em;
}

#info {
    display: none;
    position: fixed;
    top: 5px;
    left: 5px;
}


</style>
