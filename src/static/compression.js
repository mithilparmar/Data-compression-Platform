(function (window, document, undefined) {
  // code that should be taken care of right away

  window.onload = init;

  function init() {
    // the code to be called when the dom has loaded
    // #document has its nodes
    let inputform = document.getElementById("inputform");
    inputform && inputform.addEventListener("submit", async (e) => {
      e.preventDefault();
      let file = document.getElementById("inputfile").files[0];
      let model = document.getElementById("selectmodel").value;
      
      console.log(model);
      console.log(file);

      formElem = new FormData();
      formElem.append("model",model)
      formElem.append("file",file)
      
      $("#loadingoverlay").fadeIn();
      console.log("modal loaded")
      response = fetch("/compress_video",{
        method: "POST",
        body:
          formElem
      }).then(res =>{
      return res.blob()
    }).then(res =>{
      const url = window.URL.createObjectURL(res);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      // the filename you want
      a.download = 'compressed_video.mp4';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);

      $("#loadingoverlay").fadeOut();
        })
      })
  let outputform = document.getElementById("outputform");
  outputform && outputform.addEventListener("submit", async (e) => {
    e.preventDefault();
    let file = document.getElementById("inputfile").files[0];
    let model = document.getElementById("selectmodel").value;
      formElem = new FormData();
      formElem.append("model",model)
      formElem.append("file",file)
    
    $("#loadingoverlay").fadeIn();
    console.log("modal loaded");
    response = fetch("/decompress_video",{
      method: "POST",
      body:formElem

    }).then(res =>{
    return res.blob()
  }).then(res =>{
    const url = window.URL.createObjectURL(res);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    // the filename you want
    a.download = 'compressed_video.mp4';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);

    $("#loadingoverlay").fadeOut();
      })
    })
  }
})(window, document, undefined);
