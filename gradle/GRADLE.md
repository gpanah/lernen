[Gradle Helloworld](https://spring.io/guides/gs/gradle/#_find_out_what_gradle_can_do)

## Installation

- I've copied the files to `/opt/gradle-version/` where version is the version of Gradle
- Global configuration lives in `~/.gradle/gradle.properties`, which is really just properties that gradle can leverage at execution time
-  Also stored here are the cached dependencies (jars for Java) in `~/.gradle/caches/modules-2/files-2.1/`.  This is equivalent of `.m2/repository`

## Project specific

- `build.gradle` lives in the project root
- You can run `gradle build` from the root to execute the commands in the `build.gradle`, but this requires gradle to be installed on the users system.  If you run `gradle wrapper` it will generate some gradle files (`gradlew, gradlew.bat, and gradle/wrapper/..`).  Add all these to source control and then consumers won't have to install gradle.  In the future, then run `./gradlew build` to build.
- The first time it will need to download the gradle binaries.  If you need to do this through a proxy, it does NOT leverage the environment variables for httpProxy or httpsProxy.  Instead you need to set the following properties in `gradle.properties`:

```
systemProp.http.proxyHost=proxyURL
systemProp.http.proxyPort=proxyPort
systemProp.http.proxyUser=USER
systemProp.http.proxyPassword=PASSWORD
systemProp.https.proxyHost=proxyUrl
systemProp.https.proxyPort=proxyPort
systemProp.https.proxyUser=USER
systemProp.https.proxyPassword=PASSWORD
```
