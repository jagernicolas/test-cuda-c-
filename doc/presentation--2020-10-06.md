# Presentation: 2020-10-06

[TOC]

####  options

```
cli-test-app [--float|--int] [--width <value>] [--height <value>] [[--counter|--timer] <value>] [--record <file>] [--config <file>]
```

- `float` type float (1.0). Cannot be use with `int`. Default is false
- `int` type integer (1). Cannot be used with `float`. Default is true
- `width` width of the array. Default value is 100,
- `height` height of the array. Default value is 100.
- `counter` number of iteration. Cannot be used with `timer`. Default value is invalid.
- `timer` time to process in seconds. Cannot be used with `counter`. Default value is invalid.
- `record` this option allows to record timers to `<file>`. Default value is void. If not set, timers will be shown on screen.
- `config` load options from `<file>`. Default value is void. note: any option set from the command line will overwrite option from `<file>`.
- Ex:
  - `cli-test-app --float --width 1920 --height 1080 --timer 60 --record record.log` create an array of 1920x1080 storing 1.0, then copy back and forth between host and GPU during 60 seconds. Record timers into `record.log`

### sequential

#### cli

![image-20201006083049725](assets\image-20201006083049725.png)

#### playWithCuda()

![ROynhi8m44JhxrDCcLmXBdvG452c80U8uZLOShpHUbDmUmn2H0AgdPqdFMdQhAFeEaVp3ARfUBdX83pZn5bnPXpxmRxwydNzi65hWaSbDB6u_nfwfOHAPQNQ_3MZSq0tx7VGC9DaA2Fo1Jv4iczUTBymnKoe_5ZEmxL8IFefovCG9RlXulgcb5pmPHqlfYsvJBq3](assets\ROynhi8m44JhxrDCcLmXBdvG452c80U8uZLOShpHUbDmUmn2H0AgdPqdFMdQhAFeEaVp3ARfUBdX83pZn5bnPXpxmRxwydNzi65hWaSbDB6u_nfwfOHAPQNQ_3MZSq0tx7VGC9DaA2Fo1Jv4iczUTBymnKoe_5ZEmxL8IFefovCG9RlXulgcb5pmPHqlfYsvJBq3.png)