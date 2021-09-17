# OCR-PIL-ImageMod-fun-WIP-
{WORK IN PROGRESS OVER A LONGER PERIOD} Smaller or bigger creations/fun/tools in the fields (that I dont know much about yet) of OCR, Imaging processes and eventually forms of 'pseudo-AI's....


![equation](http://latex.codecogs.com/gif.latex?O_t%3D%5Ctext%20%7B%20Onset%20event%20at%20time%20bin%20%7D%20t)


**CHOOSE A LICENSE SOON** 

## High res image loading up with proper algorithm, settings, preset, parameters, shaders and filter options.... *WITH A NEAT ANIMATION*
:)
![Image loading animation and speed for high res](https://github.com/ChiefsBestPal/OCR-PIL-ImageMod-fun-WIP-/blob/master/readme%20showcase/AnalyzeSpeed%2Canimation%20and%20setup.gif)
## Parameters input and look of the code in the VScode terminal
![User final inputs on shaders and parameters](https://github.com/ChiefsBestPal/OCR-PIL-ImageMod-fun-WIP-/blob/master/readme%20showcase/Screenshot%202021-09-17%20010401.png)


# | LATEST ALGORITHM ADDED | 
CIEDE2000
Since the 1994 definition did not adequately resolve the perceptual uniformity issue, the CIE refined their definition, adding five corrections:[17][18]

A hue rotation term (RT), to deal with the problematic blue region (hue angles in the neighborhood of 275°):[19]
Compensation for neutral colors (the primed values in the L*C*h differences)
Compensation for lightness (SL)
Compensation for chroma (SC)
Compensation for hue (SH)
${\displaystyle \Delta E_{00}^{*}={\sqrt {\left({\frac {\Delta L'}{k_{L}S_{L}}}\right)^{2}+\left({\frac {\Delta C'}{k_{C}S_{C}}}\right)^{2}+\left({\frac {\Delta H'}{k_{H}S_{H}}}\right)^{2}+R_{T}{\frac {\Delta C'}{k_{C}S_{C}}}{\frac {\Delta H'}{k_{H}S_{H}}}}}}{\displaystyle \Delta E_{00}^{*}={\sqrt {\left({\frac {\Delta L'}{k_{L}S_{L}}}\right)^{2}+\left({\frac {\Delta C'}{k_{C}S_{C}}}\right)^{2}+\left({\frac {\Delta H'}{k_{H}S_{H}}}\right)^{2}+R_{T}{\frac {\Delta C'}{k_{C}S_{C}}}{\frac {\Delta H'}{k_{H}S_{H}}}}}}$
Note: The formulae below should use degrees rather than radians; the issue is significant for RT.
The kL, kC, and kH are usually unity.
{\displaystyle \Delta L^{\prime }=L_{2}^{*}-L_{1}^{*}}{\displaystyle \Delta L^{\prime }=L_{2}^{*}-L_{1}^{*}}
{\displaystyle {\bar {L}}={\frac {L_{1}^{*}+L_{2}^{*}}{2}}\quad {\bar {C}}={\frac {C_{1}^{*}+C_{2}^{*}}{2}}}{\displaystyle {\bar {L}}={\frac {L_{1}^{*}+L_{2}^{*}}{2}}\quad {\bar {C}}={\frac {C_{1}^{*}+C_{2}^{*}}{2}}}
{\displaystyle a_{1}^{\prime }=a_{1}^{*}+{\frac {a_{1}^{*}}{2}}\left(1-{\sqrt {\frac {{\bar {C}}^{7}}{{\bar {C}}^{7}+25^{7}}}}\right)\quad a_{2}^{\prime }=a_{2}^{*}+{\frac {a_{2}^{*}}{2}}\left(1-{\sqrt {\frac {{\bar {C}}^{7}}{{\bar {C}}^{7}+25^{7}}}}\right)}{\displaystyle a_{1}^{\prime }=a_{1}^{*}+{\frac {a_{1}^{*}}{2}}\left(1-{\sqrt {\frac {{\bar {C}}^{7}}{{\bar {C}}^{7}+25^{7}}}}\right)\quad a_{2}^{\prime }=a_{2}^{*}+{\frac {a_{2}^{*}}{2}}\left(1-{\sqrt {\frac {{\bar {C}}^{7}}{{\bar {C}}^{7}+25^{7}}}}\right)}
{\displaystyle {\bar {C}}^{\prime }={\frac {C_{1}^{\prime }+C_{2}^{\prime }}{2}}{\mbox{ and }}\Delta {C'}=C'_{2}-C'_{1}\quad {\mbox{where }}C_{1}^{\prime }={\sqrt {a_{1}^{'^{2}}+b_{1}^{*^{2}}}}\quad C_{2}^{\prime }={\sqrt {a_{2}^{'^{2}}+b_{2}^{*^{2}}}}\quad }{\displaystyle {\bar {C}}^{\prime }={\frac {C_{1}^{\prime }+C_{2}^{\prime }}{2}}{\mbox{ and }}\Delta {C'}=C'_{2}-C'_{1}\quad {\mbox{where }}C_{1}^{\prime }={\sqrt {a_{1}^{'^{2}}+b_{1}^{*^{2}}}}\quad C_{2}^{\prime }={\sqrt {a_{2}^{'^{2}}+b_{2}^{*^{2}}}}\quad }
{\displaystyle h_{1}^{\prime }={\text{atan2}}(b_{1}^{*},a_{1}^{\prime })\mod 360^{\circ },\quad h_{2}^{\prime }={\text{atan2}}(b_{2}^{*},a_{2}^{\prime })\mod 360^{\circ }}{\displaystyle h_{1}^{\prime }={\text{atan2}}(b_{1}^{*},a_{1}^{\prime })\mod 360^{\circ },\quad h_{2}^{\prime }={\text{atan2}}(b_{2}^{*},a_{2}^{\prime })\mod 360^{\circ }}
Note: The inverse tangent (tan−1) can be computed using a common library routine atan2(b, a′) which usually has a range from −π to π radians; color specifications are given in 0 to 360 degrees, so some adjustment is needed. The inverse tangent is indeterminate if both a′ and b are zero (which also means that the corresponding C′ is zero); in that case, set the hue angle to zero. See Sharma 2005, eqn. 7.
{\displaystyle \Delta h'={\begin{cases}h_{2}^{\prime }-h_{1}^{\prime }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|\leq 180^{\circ }\\h_{2}^{\prime }-h_{1}^{\prime }+360^{\circ }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{2}^{\prime }\leq h_{1}^{\prime }\\h_{2}^{\prime }-h_{1}^{\prime }-360^{\circ }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{2}^{\prime }>h_{1}^{\prime }\end{cases}}}{\displaystyle \Delta h'={\begin{cases}h_{2}^{\prime }-h_{1}^{\prime }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|\leq 180^{\circ }\\h_{2}^{\prime }-h_{1}^{\prime }+360^{\circ }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{2}^{\prime }\leq h_{1}^{\prime }\\h_{2}^{\prime }-h_{1}^{\prime }-360^{\circ }&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{2}^{\prime }>h_{1}^{\prime }\end{cases}}}
Note: When either C′1 or C′2 is zero, then Δh′ is irrelevant and may be set to zero. See Sharma 2005, eqn. 10.
{\displaystyle \Delta H^{\prime }=2{\sqrt {C_{1}^{\prime }C_{2}^{\prime }}}\sin(\Delta h^{\prime }/2),\quad {\bar {H}}^{\prime }={\begin{cases}(h_{1}^{\prime }+h_{2}^{\prime })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|\leq 180^{\circ }\\(h_{1}^{\prime }+h_{2}^{\prime }+360^{\circ })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{1}^{\prime }+h_{2}^{\prime }<360^{\circ }\\(h_{1}^{\prime }+h_{2}^{\prime }-360^{\circ })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{1}^{\prime }+h_{2}^{\prime }\geq 360^{\circ }\end{cases}}}{\displaystyle \Delta H^{\prime }=2{\sqrt {C_{1}^{\prime }C_{2}^{\prime }}}\sin(\Delta h^{\prime }/2),\quad {\bar {H}}^{\prime }={\begin{cases}(h_{1}^{\prime }+h_{2}^{\prime })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|\leq 180^{\circ }\\(h_{1}^{\prime }+h_{2}^{\prime }+360^{\circ })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{1}^{\prime }+h_{2}^{\prime }<360^{\circ }\\(h_{1}^{\prime }+h_{2}^{\prime }-360^{\circ })/2&\left|h_{1}^{\prime }-h_{2}^{\prime }\right|>180^{\circ },h_{1}^{\prime }+h_{2}^{\prime }\geq 360^{\circ }\end{cases}}}
Note: When either C′1 or C′2 is zero, then H′ is h′1+h′2 (no divide by 2; essentially, if one angle is indeterminate, then use the other angle as the average; relies on indeterminate angle being set to zero). See Sharma 2005, eqn. 7 and p. 23 stating most implementations on the internet at the time had "an error in the computation of average hue".
{\displaystyle T=1-0.17\cos({\bar {H}}^{\prime }-30^{\circ })+0.24\cos(2{\bar {H}}^{\prime })+0.32\cos(3{\bar {H}}^{\prime }+6^{\circ })-0.20\cos(4{\bar {H}}^{\prime }-63^{\circ })}{\displaystyle T=1-0.17\cos({\bar {H}}^{\prime }-30^{\circ })+0.24\cos(2{\bar {H}}^{\prime })+0.32\cos(3{\bar {H}}^{\prime }+6^{\circ })-0.20\cos(4{\bar {H}}^{\prime }-63^{\circ })}
{\displaystyle S_{L}=1+{\frac {0.015\left({\bar {L}}-50\right)^{2}}{\sqrt {20+{\left({\bar {L}}-50\right)}^{2}}}}\quad S_{C}=1+0.045{\bar {C}}^{\prime }\quad S_{H}=1+0.015{\bar {C}}^{\prime }T}{\displaystyle S_{L}=1+{\frac {0.015\left({\bar {L}}-50\right)^{2}}{\sqrt {20+{\left({\bar {L}}-50\right)}^{2}}}}\quad S_{C}=1+0.045{\bar {C}}^{\prime }\quad S_{H}=1+0.015{\bar {C}}^{\prime }T}
{\displaystyle R_{T}=-2{\sqrt {\frac {{\bar {C}}'^{7}}{{\bar {C}}'^{7}+25^{7}}}}\sin \left[60^{\circ }\cdot \exp \left(-\left[{\frac {{\bar {H}}'-275^{\circ }}{25^{\circ }}}\right]^{2}\right)\right]}{\displaystyle R_{T}=-2{\sqrt {\frac {{\bar {C}}'^{7}}{{\bar {C}}'^{7}+25^{7}}}}\sin \left[60^{\circ }\cdot \exp \left(-\left[{\frac {{\bar {H}}'-275^{\circ }}{25^{\circ }}}\right]^{2}\right)\right]}

[CIEDE2000][OP;15][TRANS;6]test_wall.jpg
![dfdfs](https://github.com/ChiefsBestPal/OCR-PIL-ImageMod-fun-WIP-/blob/master/%5BCIEDE2000%5D%5BOP%3B15%5D%5BTRANS%3B6%5Dtest_wall.jpg)
[CIEDE2000][OP;1][TRANS;2]test_wallDARKGREY.jpg
![dsasac](https://github.com/ChiefsBestPal/OCR-PIL-ImageMod-fun-WIP-/blob/master/%5BCIEDE2000%5D%5BOP%3B1%5D%5BTRANS%3B2%5Dtest_wallDARKGREY.jpg)
test_wallDARKGREYXtest_wallLIGHTGREYSuperimposed.jpg
![sadca](https://github.com/ChiefsBestPal/OCR-PIL-ImageMod-fun-WIP-/blob/master/test_wallDARKGREYXtest_wallLIGHTGREYSuperimposed.jpg)

|                |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|
