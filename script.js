let male = 0
let female = 0

const ctx = document.getElementById('chart')

const chart = new Chart(ctx, {

type: 'bar',

data: {

labels: ['Male','Female'],

datasets: [{

label: 'Predictions',

data: [male,female]

}]

}

})



function predict(){

const fileInput = document.getElementById("imageUpload")

const file = fileInput.files[0]

const formData = new FormData()

formData.append("file",file)

fetch("/predict",{

method:"POST",
body:formData

})

.then(res=>res.json())

.then(data=>{

document.getElementById("result").innerHTML = 
"Prediction: " + data.gender

if(data.gender==="Male"){
male++
}
else{
female++
}

chart.data.datasets[0].data=[male,female]

chart.update()

})

}