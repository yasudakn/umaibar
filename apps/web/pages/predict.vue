<template>
  <div id="container">
    <div class="title">
      <h1>うまい棒 分類</h1>
    </div>
    <div class="testdata">
      <div class="target">
        <div><label>テストデータのサンプル</label></div>
        <div v-for="(item, i) in list" v-bind:key="i" class="samples">
          <img :src="item.url" @click="handleImageSelect">
        </div>
        <label>
          <b-form-file v-model="file" :state="Boolean(file)" @change="handleFileUpload" placeholder="またはファイルを選択"/>
        </label>
      </div>
      <div class="selected">
        <label>選択中</label>
        <img v-if="preview" :src="preview" />
      </div>
      <div class="predict">
        <b-button @click="submitFile">推論する</b-button>
      </div>
    </div>
    <hr/>
    <div class="results" v-if="results">
      <div class="top_k">
        <div class="result_title">結果(Top5)</div>
        <div v-for="(result, i) in results" v-bind:key="i">
	  <div v-if="i == 0" class="top_1">
            <label>{{ result }}</label>
	  </div>
	  <div v-else class="result">
            <label>{{ result }}</label>
	  </div>
        </div>
      </div>
      <div class="cam_image" v-if="gradcam_image">
        <img v-bind:src="'data:image/png;base64,'+gradcam_image" />
      </div>
      <div class="cam_image" v-if="guided_image">
        <img v-bind:src="'data:image/png;base64,'+guided_image" />
      </div>
    </div>
  </div>
</template>
vv
<script>
import axios from 'axios';

var url = 'http://localhost:13001/predict';

export default {
  name: 'container',
    
  /*
    Defines the data used by the component
  */
  data: function(){
    return {
      file: null,
      results: null,
      gradcam_image: null,
      guided_image: null,
      preview: null,
      list: []
    }
  },
  created (){
    let self = this;
    let data = [
      	    {'url':'http://localhost:13000/P1040758_half.JPG'},
	    {'url':'http://localhost:13000/P1040751_half.JPG'},
	    {'url':'http://localhost:13000/P1040932_half.JPG'},
	    {'url':'http://localhost:13000/P1040316_half.JPG'},
	    {'url':'http://localhost:13000/P1040929_half.JPG'},
            {'url':'http://localhost:13000/curl_cheese.jpg'}
	    ];
    self.list = data;
  },
  methods: {
    /*
      Submits the file to the server
    */
    submitFile(){
      let self = this;
      let formData = new FormData();
      
      console.log(self.file);
      formData.append('file', self.file);

      axios.post(
        url,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      ).then(function(response){
        console.log(response);
	var scores = response.data.scores;
	var top5 = response.data.predict_top5.map((v, i) => v[1] + '(' + Math.floor(scores[v[0]] * 100) / 100 + ')');
	console.log(top5);
	self.results = top5.map((v, i) => v);  // response.data.predict_top5[0][1] + '(' + top_score + ')';
	self.gradcam_image = response.data.gradcam_image;
	self.guided_image = response.data.guided_gradcam_image;
      }).catch(function(error){
        console.log('FAILURE!! ' + error);
      });
    },

    /*
      Handles a change on the file upload
    */
    handleFileUpload(e){
      let self = this;
      self.file = e.target.files[0];
      console.log(self.file);
      self.preview = URL.createObjectURL(self.file);
    },
    /*
      handles a click on the image file
    */
    handleImageSelect(event){
      let self = this;
      var url = event.target.src;
      var request = new XMLHttpRequest();
      request.open('GET', url, true);
      request.responseType = 'blob';
      request.onload= function(){
	var blob = URL.createObjectURL(request.response);
        self.preview = blob;
	var reader = new FileReader();
	reader.onload = function(e) {
	  self.file = new File([e.target.result], url.split('/').pop());
	  console.log(self.file);
	};
        reader.readAsArrayBuffer(request.response);
      };
      request.send();
    }
  },
}
</script>

<style scoped>
.title, .testdata, .predict, .samples, .results, .selected {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 10px;
}

.target {
  display: inline-block;
}

.selected img, .cam_image img{
  max-width: 100%;
  max-height: 224px;
}

.top_k {
  display: block;
}

.top_1 {
  color: crimson;
}

.result_title {
  font-weight: bold;
}

.result {
  width: auto;
}

.samples, .cam_image {
  display: inline-block;
  padding: 10px;
}

.samples img{
  max-width: 100%;
  max-height: 100px;
}
</style>
