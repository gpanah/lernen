http://www.nicolasbize.com/blog/7-tips-to-speed-up-eclipse/
- General > Startup and Shutdown : remove all plugins activated on startup
- General > Editors > Text Editors > Spelling : Disable spell checking
- General > Validation > Suspend all
- Window > Customize Perspective > Remove stuff you don’t use or want (shortcut keys are your friends), same for Menu Visibility (how many times have you printed a source file…)
- Install/Update > Automatic Updates > Uncheck “Automatically find new updates”
- General > Appearance > Uncheck Enable Animations
- Stay with the default theme. Unfortunately, anything else makes it really laggy and slow.
- Java > Editor > Content Assist > disable Enable Auto Activation. Advanced > Remove all unwanted kinds

For STS specific

- Spring > Validation (disable)
- Spring > dashboard (uncheck "News Feed Updates")

STS.ini

turn on performance compiler optimizations

`-XX:+AggressiveOpts`

increase permanent generation space (where new objects are allocated)

`-XX:PermSize=512m`
`-XX:MaxPermSize=512m`

increase min and max heap sizes (which includes young and tenured generations)

`-Xms2048m`
`-Xmx2048m`

increase heap size for the young generation

`-Xmn512m`

set stack size for each thread

`-Xss2m`

tweak garbage collection

`-XX:+UseParallelOldGC`
