var mocky = require('mocky')

mocky.createServer([{
  url: '/single-file',
  method: 'post',
  headers: {'Content-Type': 'multipart/form-data'},
  res: {
        status: 200,
        headers: {'Content-Type': 'text/json', 'Access-Control-Allow-Origin': 'http://localhost:13000'},
        body: JSON.stringify({'status': 'ok'})
       }
  }]).listen(3001);
