"use strict"

$(document).ready (function(){
    $("#success-alert").hide();
 });

function showAlertAndRedirect(redirect){
    $("#success-alert").hide();
    $("#success-alert").fadeTo(2000, 500).slideUp(500, function(){
        $("#success-alert").slideUp(500);
        window.location.href = redirect;
    });
}

function load_data() {
    $("#loading").show();
    $("#options").hide();

    var dataName = prompt("Enter a name of your data: (location: './app/static/data/)'",
                            "CAPC_tokenized_temp.csv");

    dataName  = "./app/static/data/" + dataName;

    var labels = document.getElementById('labels');
    var saved_model = document.getElementById('saved_model');
    var zero_marker = document.getElementById('marker');


    $.ajax({
      type: "POST",
      url: "/",
      data: JSON.stringify(["load_data", dataName, labels.checked, saved_model.checked, zero_marker.checked]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
            $("#success-alert").text(data.msg);
            $("#loading").hide();
            $("#options").show();

            console.log(data);
            showAlertAndRedirect(data.redirect);
        }
    });
}
