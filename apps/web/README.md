# predict umaibar web app

> Nuxt.js project

## Build Setup

``` bash
# install dependencies
$ npm install # Or yarn install

# serve with hot reload at localhost:3000
$ npm run dev

# build for production and launch server
$ npm run build
$ npm start

# nginx start
$ cp -f ../nginx/sites-available/default /etc/nginx/sites-available/
$ cp -fR .nuxt /var/www/
$ service nginx restart

# generate static project
$ npm run generate
```

For detailed explanation on how things work, checkout the [Nuxt.js docs](https://github.com/nuxt/nuxt.js).

