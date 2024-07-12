# Number-classification-mnist

Ranka rašytų skaičių klasifikacijai apmokytas daugiasluoksnis perceptronas. Apmokymo bei testavimo duomenis galima rasti [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/). 10% apmokymo duomenų buvo panaudota sudarant validacijos duomenų imtį. Sukurtą modelį apibūdinančios diagramos pateiktos žemiau (Susimaišymo matricos sudarytos naudojant testavimo duomenis).
<br>
<div align="center">
  
| ![Tikslo funkcijos reikšmės priklausomybė nuo epochų skaičiaus](plots/loss_through_epochs.png) |
|:--:|
| *Tikslo funkcijos reikšmės priklausomybė nuo epochų skaičiaus* |

</div>
<br>
<div align="center">
  
| ![Tikslumo, apskaičiuoto naudojant validacijos duomenų rinkinį, reikšmės priklausomybė nuo epochų skaičiaus](plots/accuracy_through_epochs.png) |
|:--:|
| *Tikslumo, apskaičiuoto naudojant validacijos duomenų rinkinį,<br> reikšmės priklausomybė nuo epochų skaičiaus* |

</div>
<br>
<div align="center">
  
| ![Eilučių atžvilgiu normalizuota susimaišymo matrica](plots/cm_row.png) |
|:--:|
| *Eilučių atžvilgiu normalizuota susimaišymo matrica* |

</div>
<br>
<div align="center">
  
| ![Stulpelių atžvilgiu normalizuota susimaišymo matrica](plots/cm_column.png) |
|:--:|
| *Stulpelių atžvilgiu normalizuota susimaišymo matrica* |

</div>
<br>
Aplikacijos, leidžiančios išbandyti sukurtą skaičių klasifikacijos modelį, vartotojo sąsajos pavyzdys pateiktas žemiau.
<br>
<div align="center">
  
| ![Sukurtos aplikacijos vartotojo sąsajos pavyzdys](gui/gui_example.png) |
|:--:|
| *Sukurtos aplikacijos vartotojo sąsajos pavyzdys* |

</div>