package com.athenas.athenas.controller;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.dto.EmpresaDTO;
import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.service.EmpresaService;

@RestController
@RequestMapping("/api/empresa")
public class EmpresaController {
    
    private final EmpresaService empresaService;
    
    public EmpresaController(EmpresaService empresaService) {
        this.empresaService = empresaService;
    }
    
    @PostMapping("/create")
    public Empresa createEmpresa(@RequestBody EmpresaDTO empresaDTO) {
        return empresaService.createEmpresa(empresaDTO);
    }

    @GetMapping("getEmpresas")
    public List<Empresa> getAllEmpresas(){
        return empresaService.getAllEmpresas();
    }
    
    @GetMapping("/getById/{idEmpresa}")
    public Empresa getEmpresaById(@PathVariable int idEmpresa) {
        return empresaService.getEmpresaById(idEmpresa);
    }
    
    @GetMapping("/getByNome/{nomeEmpresa}")
    public Empresa getEmpresaByNome(@PathVariable String nomeEmpresa) {
        return empresaService.getEmpresaByNome(nomeEmpresa);
    }
    
    @GetMapping("/getByCnpj/{cnpjEmpresa}")
    public Empresa getEmpresaByCNPJ(@PathVariable String cnpjEmpresa) {
        return empresaService.getEmpresaByCNPJ(cnpjEmpresa);
    }
    
    @PutMapping("/update/{idEmpresa}")
    public Empresa updateEmpresa(@PathVariable long idEmpresa, @RequestBody EmpresaDTO empresaDTO) {
        return empresaService.updateEmpresa(idEmpresa, empresaDTO);
    }

    @DeleteMapping("/delete/{idEmpresa}")
    public void deleteEmpresa(@PathVariable long idEmpresa) {
        empresaService.deleteEmpresa(idEmpresa);
    }
}